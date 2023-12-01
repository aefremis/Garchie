from fetch_equity import crypto, asset
import pandas as pd
# select asset
asset_series = asset(symbol='IBM', granularity='1d', start='2023-01-01', end='2023-11-21')
print(asset_series)
raw = asset_series.fetch_asset()
raw.reset_index(inplace=True)
ts = raw['return'].copy()



class garch_model:
    """
        A class used to represent the optimal GARCH  model to be used as a volatility model to the prediction committee

        ...
        Attributes
        ----------
        ts : dataframe
            a dataframe that contains the asset information
        forecast_ahead : integer
            a number that dictates the forecasting horizon and holdout evaluation window
        fixed_window : bool
            a boolean that chooses between rolling and fixed window forecast for the evaluation
        diagnostics : bool
            a boolean that prints the diagnostic/prediction plots

        Methods
        -------
        design_garch_model()
            Designs the optimal autoregressive conditional heteroskedasticity model for a given time-series
    """
    def __init__(self, ts, forecast_ahead, fixed_window, diagnostics):
        """
        Parameters
       ----------
        ts : dataframe
            a dataframe that contains the asset information
        forecast_ahead : integer
            a number dictates the forecasting horizon and holdout evaluation window
        fixed_window : bool
            a boolean that chooses between rolling and fixed window forecast for the evaluation
        diagnostics : bool
            a boolean that prints the diagnostic/prediction plots
        """
        self.ts = ts
        self.forecast_ahead = forecast_ahead
        self.fixed_window = fixed_window
        self.diagnostics = diagnostics

    def __str__(self):
        return f"class of optimal GARCH model for the given asset"

    def fit_evaluate_garch(self, p_order, q_order, o_order, mean_type, vol_type, dist_type):
        """
        Fits a specified GARCH model and returns its performance metrics

        Parameters
        ----------
        p_order : integer
        Lag order of the symmetric innovation
        q_order : integer
        Lag order of lagged volatility
        o_order : integer
        Lag order of the asymmetric innovation
        mean_type : Name of the mean model
        vol_type : Name of the volatility model
        dist_type : Name of the error distribution

        Returns
        -------
        DataFrame :
        a dictionary of performance metrics (aic, bic, loglikelihood)
        """
        import arch
        model = arch.arch_model(y=self.ts, vol=vol_type, p=p_order, q=q_order,
                                o=o_order, mean=mean_type, dist=dist_type)
        results = model.fit(disp='off')
        return {"aic": results.aic,
                "bic": results.bic,
                "loglikelihood": results.loglikelihood}

    ## Backtesting with MAE, MSE
    def evaluate(self, observation, forecast):
        """
        Fits a specified GARCH model and returns its performance metrics

        Parameters
        ----------
        observation : array
        observed variance in hold-out period
        forecast : array
        predicted variance in hold-out period

        Returns
        -------
        List : list
        a list of accuracy metrics (mae, mse)
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        # Call sklearn function to calculate MAE
        mae = mean_absolute_error(observation, forecast)
        print('Mean Absolute Error (MAE): {:.3g}'.format(mae))
        # Call sklearn function to calculate MSE
        mse = mean_squared_error(observation, forecast)
        print('Mean Squared Error (MSE): {:.3g}'.format(mse))
        return mae, mse

    def design_garch_model(self):
        """
        Designs the optimal GARCH model for a given time-series

        Parameters
        ----------
        self : an object of class garch_model
        An object of class 'garch_model' with relevant attributes

        Returns
        -------
        conf
        a volatility based confidence interval for any given mean prediction
        """
        import numpy as np
        import arch
        from itertools import product
        from arch.__future__ import reindexing

        # Define the parameter grid
        p_values = range(1, 4)
        q_values = range(3)
        o_values = range(1)
        mean_type_values = ['constant', 'zero', 'AR']
        distribution_values = ['normal', 'skewt', 't']
        vol_values = ['GARCH', 'EGARCH']
        parameter_grid = list(product(p_values, q_values, o_values,
                                      mean_type_values,
                                      vol_values,
                                      distribution_values))

        parameter_grid = pd.DataFrame(parameter_grid, columns=['p_order', 'q_order', 'o_order',
                                                               'mean_type_values',
                                                               'vol_type_values',
                                                               'distribution_values'])
        # Initialize variables to store the best results
        best_params = None
        results_df = pd.DataFrame()

        # Perform grid search
        for i in range(parameter_grid.shape[0]):
            try:
                p_order, q_order, o_order, mean_type, vol_type, dist_type = parameter_grid.iloc[i, :]
                p_order, q_order, o_order = int(p_order), int(q_order), int(o_order)
                candidate = self.fit_evaluate_garch(p_order, q_order, o_order, mean_type, vol_type, dist_type)
                results_df = results_df.append({'aic': candidate['aic'],
                                                'sbc': candidate['bic'],
                                                'loglikelihood': candidate['loglikelihood']},
                                               ignore_index=True)
                print(f'Itteration {i} of {parameter_grid.shape[0]} ')
            except:
                continue

        # multiple ranking on performance criteria
        results_df['rank_aic'] = results_df['aic'].rank()
        results_df['rank_sbc'] = results_df['sbc'].rank()
        results_df['rank_loglik'] = results_df['loglikelihood'].rank(ascending=False)
        results_df['rank_total'] = results_df.eval('(rank_aic + rank_sbc + rank_loglik) /3 ')

        # best possible model
        best_pos = results_df['rank_total'].argmin()
        results_df.drop(['rank_aic', 'rank_sbc', 'rank_loglik', 'rank_total'], axis=1, inplace=True)
        best_p, best_q, best_o, best_mean, best_vol, best_dist = parameter_grid.iloc[best_pos,:]
        # Fit the best model to the full data
        best_p, best_q, best_o = int(best_p), int(best_q), int(best_o)
        best_model = arch.arch_model(ts, vol=best_vol, p=best_p, q=best_q, o=best_o,
                                     mean=best_mean, dist=best_dist)
        best_results = best_model.fit(disp='off')

        # Visualize results
        # best_results.plot().show()

        if self.fixed_window:
            ts.index = raw.date
            index = ts.index
            start_loc = 0
            end_loc = len(index) - self.forecast_ahead
            forecasts = {}
            for i in range(len(ts) - end_loc):
                sys.stdout.write('o')
                sys.stdout.flush()
                res = best_model.fit(first_obs=start_loc + i, last_obs=i + end_loc, disp='off')
                temp = res.forecast(horizon=1, start=end_loc).variance
                fcast = temp.iloc[i]
                forecasts[fcast.name] = fcast
            print(' Done!')
            variance_holdout = pd.DataFrame(forecasts).T
            variance_holdout = pd.Series(variance_holdout['h.1'])
        else:
            ts.index = raw.date
            index = ts.index
            start_loc = 0
            end_loc = len(index) - self.forecast_ahead
            forecasts = {}
            for i in range(len(ts) - end_loc):
                sys.stdout.write('-')
                sys.stdout.flush()
                res = best_model.fit(first_obs=start_loc, last_obs=i + end_loc, disp='off')
                temp = res.forecast(horizon=1, start=end_loc).variance
                fcast = temp.iloc[i]
                forecasts[fcast.name] = fcast
            print(' Done!')
            variance_holdout = pd.DataFrame(forecasts).T
            variance_holdout = pd.Series(variance_holdout['h.1'])
            # print(best_results.summary())

        # Backtest model with MAE, MSE
        observed_var = raw['return'].tail(self.forecast_ahead).sub(raw['return'].tail(self.forecast_ahead).mean()).pow(2)
        self.evaluate(observed_var, variance_holdout)

        if self.diagnostics:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            # Plot the actual  volatility
            bm_std = best_results.conditional_volatility
            plt.plot(raw['return'].sub(raw['return'].mean()).pow(2),
                     color='grey', alpha=0.4, label='Daily Volatility')

            # Plot EGARCH  estimated volatility
            plt.plot(bm_std ** 2, color='red', label='Best Model Volatility')
            plt.legend(loc='upper right')
            plt.show()

        if best_vol == 'EGARCH':
            forecast = best_results.forecast(horizon=self.forecast_ahead, method="bootstrap").variance.T
        else:
            forecast = best_results.forecast(horizon=self.forecast_ahead).variance.T

        conf = np.sqrt(forecast) * 1.96
        return(conf)


# sample run
'''
gg = garch_model(ts=ts, forecast_ahead=7, fixed_window=False, diagnostics=False)
print(gg)
gg.design_garch_model()
'''

