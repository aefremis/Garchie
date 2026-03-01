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
    def __init__(self, ts: pd.DataFrame, train_size: float, forecast_ahead: int, diagnostics: bool):
        """
        Parameters
        ----------
        ts : pd.DataFrame
            Input DataFrame containing 'date' and 'typical_price' columns.
        train_size : float
            Proportion of data to use for training (0 to 1).
        forecast_ahead : int
            Number of periods to forecast into the future.
        diagnostics : bool
            If True, displays diagnostic plots during execution.
        """
        self.ts = ts
        self.train_size = train_size
        self.forecast_ahead = forecast_ahead
        self.diagnostics = diagnostics

    def __str__(self):
        return f"Theta Model (Forecast: {self.forecast_ahead} steps)"

    def _get_seasonal_period(self) -> int:
        """
        Detects the dominant seasonal period (m) using Autocorrelation Function (ACF).

        Returns
        -------
        int
            Detected seasonal period. Defaults to 1 if no strong seasonality is found.
        """
        try:
            # Prepare data: use typical price
            data = self.ts['typical_price'].values
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

    def design_theta_model(self) -> pd.DataFrame:
        """
        Fits the optimal Theta model for a given time-series.

        Returns
        -------
        pd.DataFrame
            Standardized forecast DataFrame with columns:
            ['date', 'prediction', 'model_name', 'variable', 'lower_bound', 'upper_bound']
        """
        # Ensure date column is datetime and set as index for modeling/plotting
        ts_indexed = self.ts.copy()
        ts_indexed['date'] = pd.to_datetime(ts_indexed['date'])
        ts_indexed.set_index('date', inplace=True)
        
        # Set a frequency to help statsmodels project correctly.
        # We try to infer it, or set to 'W' (Weekly) since that's often the context here,
        # but 'B' (Business Days) or 'D' (Daily) might also apply based on data.
        inferred_freq = pd.infer_freq(ts_indexed.index)
        if inferred_freq is None:
            # Reindex with forward fill if necessary to establish a regular grid
            # Or simpler: create a period index if dates are too messy for statsmodels
            ts_indexed = ts_indexed.asfreq('W-FRI', method='ffill')

        split_idx = round(len(ts_indexed) * self.train_size)
        # Split using the indexed series
        train = ts_indexed.iloc[:split_idx]['typical_price']
        test = ts_indexed.iloc[split_idx:]['typical_price']
        
        # Seasonal period detection via ACF
        m = self._get_seasonal_period()
        print(f"Detected seasonal period (m): {m}")

        # Validation Model
        val_model = ThetaModel(train, period=m)
        val_res = val_model.fit()

        # Validation Predictions
        fc = val_res.forecast(len(test))
        conf_int = val_res.prediction_intervals(len(test))
        test_df = pd.DataFrame({'pred': fc.values,
                                'lower': conf_int.iloc[:, 0].values,
                                'upper': conf_int.iloc[:, 1].values})
        test_df.index = test.index

        # Retrain Best Model on Full Data for Future Forecast
        best_model = ThetaModel(ts_indexed['typical_price'], period=m)
        best_res = best_model.fit()

        forecast = best_res.forecast(self.forecast_ahead)
        conf_int_forecast = best_res.prediction_intervals(self.forecast_ahead)
        
        pred_df = pd.DataFrame({
            'prediction': forecast.values,
            'lower_bound': conf_int_forecast.iloc[:, 0].values,
            'upper_bound': conf_int_forecast.iloc[:, 1].values
        })
        
        # Manually create the future dates since Statsmodels might drop them
        # if the frequency is complex
        future_dates = pd.date_range(start=ts_indexed.index[-1], periods=self.forecast_ahead + 1, freq=ts_indexed.index.freq)[1:]
        pred_df['date'] = future_dates
        
        pred_df['model_name'] = 'Theta'
        pred_df['variable'] = 'price'
        
        # Reorder columns
        pred_df = pred_df[['date', 'prediction', 'model_name', 'variable', 'lower_bound', 'upper_bound']]

        # Diagnostics Plotting
        if self.diagnostics:
            try:
                EDA.plot_mean_model_diagnostics(
                    train=train,
                    test=test,
                    validation_df=test_df,
                    forecast_df=pred_df,
                    model=None # Passing None because ThetaModel is not fully compatible with SARIMAX diagnostics internals
                )
            except Exception as e:
                print(f"Diagnostics plot encountered an error, potentially due to model format: {e}")

        return pred_df


if __name__ == "__main__":
    # select asset
    asset_series = asset(symbol='VUSA.AS',granularity='1wk', start= '2023-01-01', end= '2026-01-01')
    print(asset_series)
    raw = asset_series.fetch_asset()
    raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
    ts = raw[['date', 'typical_price']].copy()

    tm = theta_model(ts=ts, train_size=0.8, forecast_ahead=10, diagnostics=True)
    print(tm)
    pred_df = tm.design_theta_model()
    print(pred_df)
