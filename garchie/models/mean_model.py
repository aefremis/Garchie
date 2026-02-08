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
    def __init__(self, ts: pd.DataFrame, train_size: float, forecast_ahead: int, stationarity: bool, diagnostics: bool):
        """
        Parameters
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
        """
        self.ts = ts
        self.train_size = train_size
        self.forecast_ahead = forecast_ahead
        self.stationarity = stationarity
        self.diagnostics = diagnostics

    def __str__(self):
        return f"ARIMA/SARIMA Mean Model (Forecast: {self.forecast_ahead} steps)"

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

    def design_mean_model(self) -> pd.DataFrame:
        """
        Fits the optimal AutoRegressive model for a given time-series.

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

        
        split_idx = round(len(ts_indexed)* self.train_size)
        # Split using the indexed series
        train = ts_indexed.iloc[:split_idx]['typical_price']
        test = ts_indexed.iloc[split_idx:]['typical_price']
        
        # d term estimation
        kpss_diffs = pmdarima.arima.ndiffs(train, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = pmdarima.arima.ndiffs(train, alpha=0.05, test='adf', max_d=6)
        n_diffs = max(adf_diffs, kpss_diffs)
        print(f"Estimated differencing term: {n_diffs}")

        # Seasonal period detection via ACF
        m = self._get_seasonal_period()
        print(f"Detected seasonal period (m): {m}")

        # Auto-ARIMA
        model = pmdarima.auto_arima(train,
                                    d=n_diffs,
                                    m=m,
                                    seasonal=True if m > 1 else False,
                                    start_p=1, start_q=1,
                                    max_p=5, max_q=5,
                                    trace=False,
                                    error_action='ignore',  
                                    suppress_warnings=True,
                                    stepwise=True)

        # Validation Predictions
        fc, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
        test_df = pd.DataFrame({'pred': fc,
                                'lower': conf_int[:, 0],
                                'upper': conf_int[:, 1]})
        test_df.set_index(test.index, inplace=True)

        # Retrain Best Model on Full Data for Future Forecast
        # Using SARIMAX with order found by auto_arima
        # We pass the full series with DatetimeIndex
        best_model = SARIMAX(ts_indexed['typical_price'],
                            order=model.order,
                            seasonal_order=model.seasonal_order).fit(disp=False)

        forecast = best_model.get_forecast(steps=self.forecast_ahead)
        pred_df = forecast.conf_int()
        pred_df['prediction'] = forecast.predicted_mean
        pred_df.columns = ['lower_bound', 'upper_bound', 'prediction']
        
        
        pred_df.reset_index(inplace=True)
        # Rename column to 'date' if it was named 'index'
        if 'index' in pred_df.columns:
            pred_df.rename(columns={'index': 'date'}, inplace=True)
        
        pred_df['model_name'] = 'ARIMA'
        pred_df['variable'] = 'price'
        
        # Reorder columns
        pred_df = pred_df[['date', 'prediction', 'model_name', 'variable', 'lower_bound', 'upper_bound']]

        # Diagnostics Plotting
        if self.diagnostics:
            EDA.plot_mean_model_diagnostics(
                train=train,
                test=test,
                validation_df=test_df,
                forecast_df=pred_df,
                model=best_model # Passing the auto_arima model for diagnostics
            )

        return pred_df

if __name__ == "__main__":
    # select asset
    asset_series = asset(symbol='VUSA.AS',granularity='1wk', start= '2023-01-01', end= '2026-01-01')
    print(asset_series)
    raw = asset_series.fetch_asset()
    raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
    ts = raw[['date', 'typical_price']].copy()

    mm = mean_model(ts=ts, train_size=0.8, forecast_ahead=10, stationarity=True, diagnostics=True)
    print(mm)
    pred_df = mm.design_mean_model()
    print(pred_df)