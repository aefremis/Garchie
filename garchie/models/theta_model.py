from garchie.data import crypto, asset, commodity
from garchie.eda import EDA
import pandas as pd
import numpy as np
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.stattools import acf

class theta_model:
    """
    Implements a Theta model for time series mean forecasting.
    
    Attributes
    ----------
    ts : pd.DataFrame
        Input DataFrame containing 'date' and 'typical_price' columns.
    train_size : float
        Proportion of data to use for training (0 to 1).
    forecast_ahead : int
        Number of periods to forecast into the future.
    diagnostics : bool
        If True, displays diagnostic plots during execution.

    Methods
    -------
    design_theta_model()
        Fits the Theta model and returns a forecast DataFrame.
    """
    def __init__(self, ts: pd.DataFrame, forecast_ahead: int, diagnostics: bool):
        """
        Parameters
        ----------
        ts : pd.DataFrame
            Input DataFrame containing 'date' and 'typical_price' columns.
        forecast_ahead : int
            Number of periods to forecast into the future.
        diagnostics : bool
            If True, displays diagnostic plots during execution.
        """
        self.ts = ts
        self.forecast_ahead = forecast_ahead
        self.diagnostics = diagnostics

    def __str__(self):
        return f"Theta Model (Forecast: {self.forecast_ahead} steps)"

    def _get_seasonal_period(self, data: pd.Series) -> int:
        """
        Detects the dominant seasonal period (m) using Autocorrelation Function (ACF).

        Returns
        -------
        int
            Detected seasonal period. Defaults to 1 if no strong seasonality is found.
        """
        try:
            # Prepare data: use typical price
            n = len(data)
            
            # Detrend the data (remove linear trend)
            x = np.arange(n)
            p = np.polyfit(x, data, 1)
            detrended = data - np.polyval(p, x)
            
            # Compute ACF
            acf_vals = acf(detrended, nlags=n//2, fft=True)
            
            # Find the peak in ACF (ignoring very short-term correlations at lags < 5)
            period = int(np.argmax(acf_vals[5:]) + 5)
            
            # Sanity checks for common periods to snap to standard values
            if 4 <= period <= 6: period = 5      # Weekly (business days)
            elif 6 <= period <= 8: period = 7    # Weekly (calendar days)
            elif 11 <= period <= 13: period = 12 # Monthly
            elif 50 <= period <= 54: period = 52 # Yearly (weeks)
            elif 28 <= period <= 31: period = 30 # Monthly (days)
            elif 360 <= period <= 370: period = 365 # Yearly (days)
            
            return period
            
        except Exception as e:
            print(f"Seasonality detection failed: {e}. Defaulting to m=1.")
            return 1

    def design_theta_model(self, validation_steps: int) -> pd.DataFrame:
        """
        Fits the optimal Theta model for a given time-series, generating both validation and future forecasts.

        Parameters
        ----------
        validation_steps : int
            Number of steps to use for validation at the end of the input time series.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing both validation and future predictions with an additional 'type' column.
            Columns: ['date', 'predicted_values', 'lower_bound', 'upper_bound', 'model_name', 'variable', 'type']
        """
        # Ensure date column is datetime and set as index for modeling/plotting
        ts_indexed = self.ts.copy()
        ts_indexed['date'] = pd.to_datetime(ts_indexed['date'])
        ts_indexed.set_index('date', inplace=True)
        
        # Set a frequency to help statsmodels project correctly.
        inferred_freq = pd.infer_freq(ts_indexed.index)
        if inferred_freq is None:
            ts_indexed = ts_indexed.asfreq('W-FRI', method='ffill')

        # --- Validation Predictions ---
        train_validation = ts_indexed.iloc[:-validation_steps]['typical_price']
        test_validation = ts_indexed.iloc[-validation_steps:]['typical_price']
        
        m_validation = self._get_seasonal_period(data=train_validation)
        print(f"Detected seasonal period for validation (m): {m_validation}")

        val_model = ThetaModel(train_validation, period=m_validation)
        val_res = val_model.fit()

        fc_val = val_res.forecast(len(test_validation))
        conf_int_val = val_res.prediction_intervals(len(test_validation))
        
        validation_pred_df = pd.DataFrame({
            'predicted_values': fc_val.values,
            'lower_bound': conf_int_val.iloc[:, 0].values,
            'upper_bound': conf_int_val.iloc[:, 1].values
        })
        validation_pred_df['date'] = test_validation.index
        validation_pred_df['model_name'] = 'Theta'
        validation_pred_df['variable'] = 'price'
        validation_pred_df['type'] = 'validation'

        # --- Future Forecast ---
        m_full = self._get_seasonal_period(data=ts_indexed['typical_price'])
        print(f"Detected seasonal period for full model (m): {m_full}")

        best_model = ThetaModel(ts_indexed['typical_price'], period=m_full)
        best_res = best_model.fit()

        forecast = best_res.forecast(self.forecast_ahead)
        conf_int_forecast = best_res.prediction_intervals(self.forecast_ahead)
        
        future_pred_df = pd.DataFrame({
            'predicted_values': forecast.values,
            'lower_bound': conf_int_forecast.iloc[:, 0].values,
            'upper_bound': conf_int_forecast.iloc[:, 1].values
        })
        
        future_dates = pd.date_range(start=ts_indexed.index[-1], periods=self.forecast_ahead + 1, freq=ts_indexed.index.freq)[1:]
        future_pred_df['date'] = future_dates
        
        future_pred_df['model_name'] = 'Theta'
        future_pred_df['variable'] = 'price'
        future_pred_df['type'] = 'future'
        
        # Combine and reorder
        combined_df = pd.concat([validation_pred_df, future_pred_df], ignore_index=True)
        combined_df = combined_df[['date', 'predicted_values', 'lower_bound', 'upper_bound', 'model_name', 'variable', 'type']]

        # Diagnostics Plotting
        if self.diagnostics:
            try:
                # Create a test_df for plotting that matches the structure of validation_pred_df but with original column names
                test_df_plot = validation_pred_df.rename(columns={'predicted_values': 'pred', 'lower_bound': 'lower', 'upper_bound': 'upper'})
                
                EDA.plot_mean_model_diagnostics(
                    train=train_validation,
                    test=test_validation,
                    validation_df=test_df_plot,
                    forecast_df=future_pred_df.rename(columns={'predicted_values': 'prediction'}),
                    model=None
                )
            except Exception as e:
                print(f"Diagnostics plot encountered an error: {e}")

        return combined_df


if __name__ == "__main__":
    # select asset
    asset_series = asset(symbol='VUSA.AS',granularity='1wk', start= '2023-01-01', end= '2026-01-01')
    print(asset_series)
    raw = asset_series.fetch_asset()
    raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
    ts = raw[['date', 'typical_price']].copy()

    tm = theta_model(ts=ts, forecast_ahead=10, diagnostics=True)
    print(tm)
    
    results_df = tm.design_theta_model(validation_steps=10)
    print(results_df)

    validation_results = results_df[results_df['type'] == 'validation']
    future_forecast_results = results_df[results_df['type'] == 'future']

    print("\nValidation Predictions:")
    print(validation_results)
    print("\nFuture Forecast:")
    print(future_forecast_results)
