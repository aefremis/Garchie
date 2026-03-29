from garchie.data import crypto, asset, commodity
from garchie.eda import EDA
import pandas as pd
import numpy as np
import pmdarima
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

class mean_model:
    """
    Implements an optimal AutoRegressive model (ARIMA/SARIMA) for time series mean forecasting.
    
    Attributes
    ----------
    ts : pd.DataFrame
        Input DataFrame containing 'date' and 'typical_price' columns.
    train_size : float
        Proportion of data to use for training (0 to 1).
    forecast_ahead : int
        Number of periods to forecast into the future.
    stationarity : bool
        If True, prints stationarity test results (ADF).
    diagnostics : bool
        If True, displays diagnostic plots during execution.

    Methods
    -------
    design_mean_model()
        Fits the AutoRegressive model and returns a forecast DataFrame.
    """
    def __init__(self, ts: pd.DataFrame, forecast_ahead: int, stationarity: bool, diagnostics: bool):
        """
        Parameters
        ----------
        ts : pd.DataFrame
            Input DataFrame containing 'date' and 'typical_price' columns.
        forecast_ahead : int
            Number of periods to forecast into the future.
        stationarity : bool
            If True, prints stationarity test results (ADF).
        diagnostics : bool
            If True, displays diagnostic plots during execution.
        """
        self.ts = ts
        self.forecast_ahead = forecast_ahead
        self.stationarity = stationarity
        self.diagnostics = diagnostics

    def __str__(self):
        return f"ARIMA/SARIMA Mean Model (Forecast: {self.forecast_ahead} steps)"

    def _get_seasonal_period(self, data: pd.Series) -> int:
        """
        Detects the dominant seasonal period (m) using Autocorrelation Function (ACF).

        Parameters
        ----------
        data : pd.Series
            The time series data to analyze for seasonality.

        Returns
        -------
        int
            Detected seasonal period. Defaults to 1 if no strong seasonality is found.
        """
        try:
            # Prepare data: use typical price (assuming data is already 'typical_price' series)
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

    def design_mean_model(self, validation_steps: int) -> pd.DataFrame:
        """
        Fits the optimal AutoRegressive model for a given time-series, generating both
        validation predictions and future forecasts.

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

        if self.stationarity:
            # stationarity -Dickey Fuller (add also kpss and decide on optimal diffs)
            # p value < 0.05 -- stationary time series
            dftest = adfuller(ts_indexed['typical_price'])
            print('\nResults of Dickey-Fuller Test: \n')

            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value',
                                                     '#Lags Used', 'Number of Observations Used'])
            for key, value in dftest[4].items():
                dfoutput['Critical Value (%s)' % key] = value
            print(dfoutput)

        # Split data for validation
        train_validation = ts_indexed.iloc[:-validation_steps]['typical_price']
        test_validation = ts_indexed.iloc[-validation_steps:]['typical_price']

        # d term estimation for validation model
        kpss_diffs = pmdarima.arima.ndiffs(train_validation, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = pmdarima.arima.ndiffs(train_validation, alpha=0.05, test='adf', max_d=6)
        n_diffs = max(adf_diffs, kpss_diffs)
        # print(f"Estimated differencing term for validation model: {n_diffs}")

        # Seasonal period detection via ACF for validation model
        m_validation = self._get_seasonal_period(data=train_validation) # Modified _get_seasonal_period to accept data
        # print(f"Detected seasonal period (m) for validation model: {m_validation}")

        # Auto-ARIMA for validation
        validation_model = pmdarima.auto_arima(train_validation,
                                    d=n_diffs,
                                    m=m_validation,
                                    seasonal=True if m_validation > 1 else False,
                                    start_p=1, start_q=1,
                                    max_p=5, max_q=5,
                                    trace=False,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)

        # Validation Predictions
        fc_val, conf_int_val = validation_model.predict(n_periods=len(test_validation), return_conf_int=True)
        validation_pred_df = pd.DataFrame({'predicted_values': fc_val, # Renamed column
                                'lower_bound': conf_int_val[:, 0],
                                'upper_bound': conf_int_val[:, 1]})
        validation_pred_df.set_index(test_validation.index, inplace=True)

        validation_pred_df.reset_index(inplace=True)
        validation_pred_df.rename(columns={'index': 'date'}, inplace=True) # Ensure 'date' column name
        validation_pred_df['model_name'] = 'ARIMA'
        validation_pred_df['variable'] = 'price'
        validation_pred_df['type'] = 'validation' # Added type column
        validation_pred_df = validation_pred_df[['date', 'predicted_values', 'lower_bound', 'upper_bound', 'model_name', 'variable', 'type']]


        # Retrain Best Model on Full Data for Future Forecast
        # d term estimation for full model
        kpss_diffs_full = pmdarima.arima.ndiffs(ts_indexed['typical_price'], alpha=0.05, test='kpss', max_d=6)
        adf_diffs_full = pmdarima.arima.ndiffs(ts_indexed['typical_price'], alpha=0.05, test='adf', max_d=6)
        n_diffs_full = max(adf_diffs_full, kpss_diffs_full)
        # print(f"Estimated differencing term for full model: {n_diffs_full}")

        # Seasonal period detection via ACF for full model
        m_full = self._get_seasonal_period(data=ts_indexed['typical_price']) # Modified _get_seasonal_period to accept data
        # print(f"Detected seasonal period (m) for full model: {m_full}")

        full_model = pmdarima.auto_arima(ts_indexed['typical_price'],
                                    d=n_diffs_full,
                                    m=m_full,
                                    seasonal=True if m_full > 1 else False,
                                    start_p=1, start_q=1,
                                    max_p=5, max_q=5,
                                    trace=False,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)


        # We pass the full series with DatetimeIndex
        best_model_full_data = SARIMAX(ts_indexed['typical_price'],
                            order=full_model.order,
                            seasonal_order=full_model.seasonal_order).fit(disp=False)

        forecast_full = best_model_full_data.get_forecast(steps=self.forecast_ahead)
        future_pred_df = forecast_full.conf_int()
        future_pred_df['predicted_values'] = forecast_full.predicted_mean # Renamed column
        future_pred_df.columns = ['lower_bound', 'upper_bound', 'predicted_values']

        # Generate future dates explicitly
        last_date = ts_indexed.index[-1]
        freq = pd.infer_freq(ts_indexed.index)
        if freq is None:
            # Fallback heuristic if frequency cannot be inferred
            if len(ts_indexed.index) > 1:
                diff = ts_indexed.index[-1] - ts_indexed.index[-2]
                if diff.days <= 2: freq = 'D' # daily or business day
                elif diff.days <= 7: freq = 'W' # weekly
                else: freq = 'D' # Default to daily if no clear pattern
            else:
                freq = 'D' # Default for single point

        future_dates = pd.date_range(start=last_date, periods=self.forecast_ahead + 1, freq=freq)[1:]
        future_pred_df['date'] = future_dates.to_numpy()
        future_pred_df['model_name'] = 'ARIMA'
        future_pred_df['variable'] = 'price'
        future_pred_df['type'] = 'future' # Added type column

        # Reorder columns
        future_pred_df = future_pred_df[['date', 'predicted_values', 'lower_bound', 'upper_bound', 'model_name', 'variable', 'type']]

        # Diagnostics Plotting (can be adapted to show both or split)
        if self.diagnostics:
            EDA.plot_mean_model_diagnostics(
                train=train_validation,
                test=test_validation,
                validation_df=validation_pred_df.rename(columns={'predicted_values': 'prediction'}), # Pass original names for plotting
                forecast_df=future_pred_df.rename(columns={'predicted_values': 'prediction'}),      # Pass original names for plotting
                model=best_model_full_data
            )

        return pd.concat([validation_pred_df, future_pred_df], ignore_index=True) # Return combined DataFrame

if __name__ == "__main__":
    # select asset
    asset_series = asset(symbol='VUSA.AS',granularity='1wk', start= '2023-01-01', end= '2026-01-01')
    print(asset_series)
    raw = asset_series.fetch_asset()
    raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
    ts = raw[['date', 'typical_price']].copy()

    mm = mean_model(ts=ts, forecast_ahead=10, stationarity=True, diagnostics=True)
    print(mm)
    # Changed call to design_mean_model
    results_df = mm.design_mean_model(validation_steps=10)
    print(results_df)

    validation_results = results_df[results_df['type'] == 'validation']
    future_forecast_results = results_df[results_df['type'] == 'future']

    print("\nValidation Predictions:")
    print(validation_results)
    print("\nFuture Forecast:")
    print(future_forecast_results)