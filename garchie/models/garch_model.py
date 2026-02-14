from garchie.data import crypto, asset
from garchie.eda import EDA
import pandas as pd
import numpy as np
import arch
from itertools import product
from arch.__future__ import reindexing
import sys

class garch_model:
    """
    A class used to represent the optimal GARCH model to be used as a volatility model to the prediction committee.

    Attributes
    ----------
    ts : pd.Series
        Input Series containing asset returns, indexed by date.
    forecast_ahead : int
        Number of periods to forecast into the future.
    fixed_window : bool
        If True, uses a fixed window for evaluation. If False, uses a rolling window.
    diagnostics : bool
        If True, prints diagnostic/prediction plots.

    Methods
    -------
    design_garch_model() -> pd.DataFrame
        Designs the optimal autoregressive conditional heteroskedasticity model for a given time-series.
    """
    def __init__(self, ts: pd.Series, forecast_ahead: int, fixed_window: bool, diagnostics: bool):
        """
        Parameters
        ----------
        ts : pd.Series
            Input Series containing asset returns, indexed by date.
        forecast_ahead : int
            Number of periods to forecast into the future.
        fixed_window : bool
            If True, uses a fixed window for evaluation. If False, uses a rolling window.
        diagnostics : bool
            If True, prints diagnostic/prediction plots.
        """
        self.ts = ts.copy()
        if not isinstance(self.ts.index, pd.DatetimeIndex):
            # Try to convert if it's not already
            pass
            
        self.forecast_ahead = forecast_ahead
        self.fixed_window = fixed_window
        self.diagnostics = diagnostics

    def __str__(self):
        return f"class of optimal GARCH model for the given asset"

    def fit_evaluate_garch(self, p_order: int, q_order: int, o_order: int, mean_type: str, vol_type: str, dist_type: str) -> dict:
        """
        Fits a specified GARCH model and returns its performance metrics.

        Parameters
        ----------
        p_order : int
            Lag order of the symmetric innovation.
        q_order : int
            Lag order of lagged volatility.
        o_order : int
            Lag order of the asymmetric innovation.
        mean_type : str
            Name of the mean model (e.g., 'Constant', 'Zero', 'AR').
        vol_type : str
            Name of the volatility model (e.g., 'GARCH', 'EGARCH').
        dist_type : str
            Name of the error distribution (e.g., 'Normal', 't').

        Returns
        -------
        dict
            A dictionary of performance metrics (aic, bic, loglikelihood).
        """
        model = arch.arch_model(y=self.ts, vol=vol_type, p=p_order, q=q_order,
                                o=o_order, mean=mean_type, dist=dist_type)
        # Increase maxiter to improve convergence
        results = model.fit(disp='off', options={'maxiter': 200})
        return {"aic": results.aic,
                "bic": results.bic,
                "loglikelihood": results.loglikelihood}

    def evaluate(self, observation: pd.Series, forecast: pd.Series) -> tuple:
        """
        Evaluates the model forecast against observed data using MAE and MSE.

        Parameters
        ----------
        observation : pd.Series
            Observed variance in hold-out period.
        forecast : pd.Series
            Predicted variance in hold-out period.

        Returns
        -------
        tuple
            A tuple of accuracy metrics (mae, mse).
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        # Call sklearn function to calculate MAE
        mae = mean_absolute_error(observation, forecast)
        print('Mean Absolute Error (MAE): {:.3g}'.format(mae))
        # Call sklearn function to calculate MSE
        mse = mean_squared_error(observation, forecast)
        print('Mean Squared Error (MSE): {:.3g}'.format(mse))
        return mae, mse

    def design_garch_model(self) -> pd.DataFrame:
        """
        Designs the optimal GARCH model for a given time-series.

        Returns
        -------
        pd.DataFrame
            A standardized dataframe containing the forecasted values with columns:
            ['date', 'prediction', 'model_name', 'variable', 'lower_bound', 'upper_bound']
        """

        # Define the parameter grid
        p_values = range(1, 2)
        q_values = [1]
        o_values = range(2)
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
        results_df = pd.DataFrame()

        # Perform grid search
        # Using a subset for faster execution in this environment if needed, but keeping full loop
        for i in range(parameter_grid.shape[0]):
            try:
                p_order, q_order, o_order, mean_type, vol_type, dist_type = parameter_grid.iloc[i, :]
                p_order, q_order, o_order = int(p_order), int(q_order), int(o_order)
                candidate = self.fit_evaluate_garch(p_order, q_order, o_order, mean_type, vol_type, dist_type)
                global_fit_dic = {'aic': candidate['aic'],
                                  'sbc': candidate['bic'],
                                  'loglikelihood': candidate['loglikelihood']
                                 }
                temp_df = pd.DataFrame.from_dict([global_fit_dic])
                results_df = pd.concat([results_df,temp_df])
                # print(f'Itteration {i} of {parameter_grid.shape[0]} ')
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
        best_model = arch.arch_model(self.ts, vol=best_vol, p=best_p, q=best_q, o=best_o,
                                     mean=best_mean, dist=best_dist)
        # Increase maxiter for final fit
        best_results = best_model.fit(disp='off', options={'maxiter': 200})

        # Rolling / Fixed Window Forecast Logic
        if self.fixed_window:
            index = self.ts.index
            start_loc = 0
            end_loc = len(index) - self.forecast_ahead
            forecasts = {}
            for i in range(len(self.ts) - end_loc):
                sys.stdout.write('o')
                sys.stdout.flush()
                # Increase maxiter for rolling forecast fits
                res = best_model.fit(first_obs=start_loc + i, last_obs=i + end_loc, disp='off', options={'maxiter': 200})
                temp = res.forecast(horizon=1, start=end_loc).variance
                fcast = temp.iloc[i]
                forecasts[fcast.name] = fcast
            print(' Done!')
            variance_holdout = pd.DataFrame(forecasts).T
            variance_holdout = pd.Series(variance_holdout['h.1'])
        else:
            index = self.ts.index
            start_loc = 0
            end_loc = len(index) - self.forecast_ahead
            forecasts = {}
            for i in range(len(self.ts) - end_loc):
                sys.stdout.write('-')
                sys.stdout.flush()
                # Increase maxiter for rolling forecast fits
                res = best_model.fit(first_obs=start_loc, last_obs=i + end_loc, disp='off', options={'maxiter': 200})
                temp = res.forecast(horizon=1, start=end_loc).variance
                fcast = temp.iloc[i]
                forecasts[fcast.name] = fcast
            print(' Done!')
            variance_holdout = pd.DataFrame(forecasts).T
            variance_holdout = pd.Series(variance_holdout['h.1'])

        # Backtest model with MAE, MSE
        observed_var = self.ts.tail(self.forecast_ahead).sub(self.ts.tail(self.forecast_ahead).mean()).pow(2)
        
        self.evaluate(observed_var, variance_holdout)

        if self.diagnostics:
            EDA.plot_garch_diagnostics(self.ts, best_results)

        if best_vol == 'EGARCH':
            try:
                forecast = best_results.forecast(horizon=self.forecast_ahead, method="bootstrap").variance.T
            except ValueError:
                # Fallback to simulation if dataset is too small for bootstrap (< 100 obs)
                print("Warning: Bootstrap forecast failed (likely insufficient data). Falling back to simulation.")
                forecast = best_results.forecast(horizon=self.forecast_ahead, method="simulation").variance.T
        else:
            forecast = best_results.forecast(horizon=self.forecast_ahead).variance.T

        # Standardize return structure
        pred_df = pd.DataFrame()
        
        # Robust Future Date Generation
        last_date = self.ts.index[-1]
        freq = pd.infer_freq(self.ts.index)
        
        if freq is None:
            # Heuristic: infer from last two points
            if len(self.ts.index) > 1:
                diff = self.ts.index[-1] - self.ts.index[-2]
                if diff.days >= 28 and diff.days <= 31: freq = 'ME' # Monthly End
                elif diff.days == 7: freq = 'W'
                elif diff.days == 1: freq = 'D'
                elif diff.days <= 3: freq = 'B' # Assume Business Day if small gap
                else: freq = 'D' # Default to daily
            else:
                freq = 'D' # Fallback for single point

        future_dates = pd.date_range(start=last_date, periods=self.forecast_ahead + 1, freq=freq)[1:]
        pred_df['date'] = future_dates
        
        volatility = np.sqrt(forecast.values.flatten())
        
        pred_df['prediction'] = volatility
        pred_df['model_name'] = 'GARCH'
        pred_df['variable'] = 'volatility'
        pred_df['lower_bound'] = np.nan
        pred_df['upper_bound'] = np.nan
        
        # Reorder columns
        pred_df = pred_df[['date', 'prediction', 'model_name', 'variable', 'lower_bound', 'upper_bound']]
        
        return(pred_df)


if __name__ == "__main__":
    # select asset
    asset_series = asset(symbol='VUSA.AS',granularity='1wk', start= '2020-01-01', end= '2026-01-01')
    print(asset_series)
    raw = asset_series.fetch_asset()
    raw.reset_index(inplace=True)
    # Ensure date is datetime
    raw['date'] = pd.to_datetime(raw['date'])
    raw.set_index('date', inplace=True)
    ts = raw['return'].copy()

    gg = garch_model(ts=ts, forecast_ahead=7, fixed_window=False, diagnostics=True)
    print(gg)
    res = gg.design_garch_model()
    print(res)
