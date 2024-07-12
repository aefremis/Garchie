from fetch_equity import crypto, asset
import pandas as pd
import numpy as np
# select asset
asset_series = asset(symbol='GPN', granularity='1d', start='2021-01-01', end='2024-07-11')
print(asset_series)
raw = asset_series.fetch_asset()
raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
ts = raw[['date', 'typical_price']].copy()


class mean_model:
    """
        A class used to represent the optimal AutoRegressive model to be used as a mean AR model the prediction committee

        ...
        Attributes
        ----------
        ts : dataframe
            a dataframe that contains the asset information
        hold_out : float
            a number that points to the size of the training set from 0 to 1 (default = 0.8)
        stationarity : bool
            a boolean that prints the stationarity tests
        diagnostics : bool
            a boolean that prints the diagnostic/prediction plots

        Methods
        -------
        design_mean_model()
            Designs the optimal AutoRegressive model for a given time-series
        """
    def __init__(self, ts, hold_out, stationarity, diagnostics):
        """
        Parameters
       ----------
        ts : dataframe
            a dataframe that contains the asset information
        hold_out : float
            a number that points to the size of the training set from 0 to 1 (default = 0.8)
        stationarity : bool
            a boolean that prints the stationarity tests
        diagnostics : bool
            a boolean that prints the diagnostic/prediction plots
        """
        self.ts = ts
        self.hold_out = hold_out
        self.stationarity = stationarity
        self.diagnostics = diagnostics

    def __str__(self):
        return f"class of optimal AutoRegressive model for the given asset "

    def design_mean_model(self):
        """Designs the optimal AutoRegressive model for a given time-series

        Parameters
        ----------
        self : an object of class mean_model
            An object of class 'mean_model' with relevant attributes

        Returns
        -------
        best_model
            a statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper model
        """
        import pmdarima
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        import matplotlib
        import matplotlib.pyplot as plt
        global model
        if self.stationarity:
            # stationarity -Dickey Fuller (add also kpss and decide on optimal diffs)
            # p value < 0.05 -- stationary time series
            dftest = adfuller(ts)
            print('\nResults of Dickey-Fuller Test: \n')

            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value',
                                                     '#Lags Used', 'Number of Observations Used'])
            for key, value in dftest[4].items():
                dfoutput['Critical Value (%s)' % key] = value
            print(dfoutput)

        # Train Test Split Index
        if self.hold_out != 0.8:
            train_size = self.hold_out
        else:
            train_size = 0.8

        split_idx = round(len(ts)* train_size)

        # Split
        train, test = ts.iloc[:split_idx]['typical_price'], ts.iloc[split_idx:]['typical_price']

        # Visualize split
        if self.diagnostics:
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.plot(train, label='Train', marker='.')
            plt.plot(test, label='Test', marker='.')
            ax.legend(bbox_to_anchor=[1, 1])
            plt.show()


        # d term
        kpss_diffs = pmdarima.arima.ndiffs(train, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = pmdarima.arima.ndiffs(train, alpha=0.05, test='adf', max_d=6)
        n_diffs = max(adf_diffs, kpss_diffs)
        print(f"Estimated differencing term: {n_diffs}")

        model = pmdarima.auto_arima(train,
                                    d=n_diffs,
                                    start_p=0,
                                    start_q=0,
                                    seasonal=True)
        if self.diagnostics:
            model.plot_diagnostics()
            plt.show()

        if self.diagnostics:
            fc, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
            pred_df = pd.DataFrame({'pred': fc,
                                    'lower': conf_int[:, 0],
                                    'upper': conf_int[:, 1]})
            pred_df.set_index(test.index, inplace=True)
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(train, label='Train', marker='.')
            ax.plot(test, label='Test', marker='.')
            ax.plot(pred_df['pred'], label='Prediction', ls='--', linewidth=3)

            ax.fill_between(x=pred_df.index, y1=pred_df['lower'], y2=pred_df['upper'], alpha=0.3)
            ax.set_title('Model Validation', fontsize=22)
            ax.legend(loc='upper left')
            fig.tight_layout()
            plt.show()

        # predict future
        best_model = SARIMAX(ts,
                            order=model.order,
                            seasonal_order=model.seasonal_order).fit()
            # best_model.summary()
        if self.diagnostics:
            best_model.plot_diagnostics()
            plt.show()
            # best_model.fittedvalues

        forecast = best_model.get_forecast(steps=7)
        pred_df = forecast.conf_int()
        pred_df['pred'] = forecast.predicted_mean
        pred_df.columns = ['lower', 'upper', 'pred']

        if self.diagnostics:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(train, label='Train', marker='.')
            ax.plot(test, label='Test', marker='.')
            ax.plot(pred_df['pred'], label='Prediction', ls='--', linewidth=3)

            ax.fill_between(x=pred_df.index, y1=pred_df['lower'], y2=pred_df['upper'], alpha=0.3)
            ax.set_title('Model Predictions', fontsize=22)
            ax.legend(loc='upper left')
            fig.tight_layout()
            plt.show()

        return(model)

mm = mean_model(ts = ts,hold_out= 0.9,stationarity= False,diagnostics=True)
print(mm)
mm.design_mean_model()

# debug predict future