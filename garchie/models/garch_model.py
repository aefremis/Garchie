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
    def __init__(self, ts: pd.Series, forecast_ahead: int, forecast_unit: str, fixed_window: bool, diagnostics: bool):
        """
        Parameters
        ----------
        ts : pd.Series
            Input Series containing asset returns, indexed by date.
        forecast_ahead : int
            Number of periods to forecast into the future.
        forecast_unit : str
            Time unit for forecasting ('days', 'weeks', 'months').
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
        self.forecast_unit = forecast_unit.lower()
        self.fixed_window = fixed_window
        self.diagnostics = diagnostics

    def __str__(self):
        return f"class of optimal GARCH model for the given asset"

    def _get_freq(self) -> str:
        """
        Maps the forecast unit to a pandas frequency string.

        Returns
        -------
        str
            Pandas frequency alias ('D', 'W', 'MS').
        """
        freq_map = {
            "days": "D",
            "weeks": "W",
            "months": "MS"  
        }
        if self.forecast_unit not in freq_map:
            raise ValueError(
                "forecast_unit must be one of: 'days', 'weeks', 'months'"
            )
        return freq_map[self.forecast_unit]

    def fit_evaluate_garch(self, ts: pd.Series, p_order: int, q_order: int, o_order: int, mean_type: str, vol_type: str, dist_type: str) -> dict:
        """
        Fits a specified GARCH model and returns its performance metrics.

        Parameters
        ----------
        ts: pd.Series
            Time series to fit the model on.
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
        model = arch.arch_model(y=ts, vol=vol_type, p=p_order, q=q_order,
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

    def _forecast_volatility(self, ts: pd.Series, best_model, forecast_horizon: int) -> pd.DataFrame:
        """
        Helper method to forecast volatility using a fitted GARCH model.

        Parameters
        ----------
        ts : pd.Series
            The time series data used for fitting.
        best_model : arch.arch_model.ARCHModelResult
            The fitted GARCH model result.
        forecast_horizon : int
            The number of steps to forecast.

        Returns
        -------
        pd.DataFrame
            A standardized dataframe containing the forecasted values.
        """
        best_vol = best_model.model.volatility.name
        if best_vol == 'EGARCH':
            try:
                forecast = best_model.forecast(horizon=forecast_horizon, method="bootstrap").variance.T
            except ValueError:
                print("Warning: Bootstrap forecast failed. Falling back to simulation.")
                forecast = best_model.forecast(horizon=forecast_horizon, method="simulation").variance.T
        else:
            forecast = best_model.forecast(horizon=forecast_horizon).variance.T

        # Standardize return structure
        pred_df = pd.DataFrame()
        
        last_date = ts.index[-1]
        freq = self._get_freq()

        future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=freq)[1:]
        pred_df['date'] = future_dates
        
        volatility = np.sqrt(forecast.values.flatten())
        
        pred_df['predicted_values'] = volatility
        pred_df['model_name'] = 'GARCH'
        pred_df['variable'] = 'volatility'
        pred_df['lower_bound'] = np.nan
        pred_df['upper_bound'] = np.nan
        
        return pred_df[['date', 'predicted_values', 'model_name', 'variable', 'lower_bound', 'upper_bound']]


    def design_garch_model(self, validation_steps: int) -> pd.DataFrame:
        """
        Designs the optimal GARCH model for a given time-series, generating both validation and future forecasts.

        Parameters
        ----------
        validation_steps : int
            Number of steps to use for validation at the end of the input time series.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing both validation and future predictions with an additional 'type' column.
        """
        # --- Validation Forecast ---
        train_validation = self.ts.iloc[:-validation_steps]
        
        # Grid search for validation model
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
        parameter_grid_df = pd.DataFrame(parameter_grid, columns=['p_order', 'q_order', 'o_order',
                                                               'mean_type_values',
                                                               'vol_type_values',
                                                               'distribution_values'])
        results_df_val = pd.DataFrame()

        for i in range(parameter_grid_df.shape[0]):
            try:
                p, q, o, mean, vol, dist = parameter_grid_df.iloc[i, :]
                p, q, o = int(p), int(q), int(o)
                candidate = self.fit_evaluate_garch(train_validation, p, q, o, mean, vol, dist)
                temp_df = pd.DataFrame.from_dict([{'aic': candidate['aic'], 'sbc': candidate['bic'], 'loglikelihood': candidate['loglikelihood']}])
                results_df_val = pd.concat([results_df_val, temp_df])
            except:
                continue

        results_df_val['rank_aic'] = results_df_val['aic'].rank()
        results_df_val['rank_sbc'] = results_df_val['sbc'].rank()
        results_df_val['rank_loglik'] = results_df_val['loglikelihood'].rank(ascending=False)
        results_df_val['rank_total'] = results_df_val.eval('(rank_aic + rank_sbc + rank_loglik) / 3')
        best_pos_val = results_df_val['rank_total'].argmin()
        best_p_val, best_q_val, best_o_val, best_mean_val, best_vol_val, best_dist_val = parameter_grid_df.iloc[best_pos_val,:]

        best_model_val = arch.arch_model(train_validation, vol=best_vol_val, p=int(best_p_val), q=int(best_q_val), o=int(best_o_val),
                                     mean=best_mean_val, dist=best_dist_val).fit(disp='off', options={'maxiter': 200})

        validation_pred_df = self._forecast_volatility(train_validation, best_model_val, validation_steps)
        validation_pred_df['type'] = 'validation'

        # --- Future Forecast ---
        results_df_future = pd.DataFrame()
        for i in range(parameter_grid_df.shape[0]):
            try:
                p, q, o, mean, vol, dist = parameter_grid_df.iloc[i, :]
                p, q, o = int(p), int(q), int(o)
                candidate = self.fit_evaluate_garch(self.ts, p, q, o, mean, vol, dist)
                temp_df = pd.DataFrame.from_dict([{'aic': candidate['aic'], 'sbc': candidate['bic'], 'loglikelihood': candidate['loglikelihood']}])
                results_df_future = pd.concat([results_df_future, temp_df])
            except:
                continue

        results_df_future['rank_aic'] = results_df_future['aic'].rank()
        results_df_future['rank_sbc'] = results_df_future['sbc'].rank()
        results_df_future['rank_loglik'] = results_df_future['loglikelihood'].rank(ascending=False)
        results_df_future['rank_total'] = results_df_future.eval('(rank_aic + rank_sbc + rank_loglik) / 3')
        best_pos_future = results_df_future['rank_total'].argmin()
        best_p_future, best_q_future, best_o_future, best_mean_future, best_vol_future, best_dist_future = parameter_grid_df.iloc[best_pos_future,:]

        best_model_future = arch.arch_model(self.ts, vol=best_vol_future, p=int(best_p_future), q=int(best_q_future), o=int(best_o_future),
                                     mean=best_mean_future, dist=best_dist_future).fit(disp='off', options={'maxiter': 200})

        future_pred_df = self._forecast_volatility(self.ts, best_model_future, self.forecast_ahead)
        future_pred_df['type'] = 'future'

        if self.diagnostics:
            EDA.plot_garch_diagnostics(self.ts, best_model_future)

        return pd.concat([validation_pred_df, future_pred_df], ignore_index=True)


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

    gg = garch_model(ts=ts, forecast_ahead=7, forecast_unit='weeks', fixed_window=False, diagnostics=True)
    print(gg)
    results_df = gg.design_garch_model(validation_steps=10)
    print(results_df)

    validation_results = results_df[results_df['type'] == 'validation']
    future_forecast_results = results_df[results_df['type'] == 'future']

    print("\nValidation Predictions:")
    print(validation_results)
    print("\nFuture Forecast:")
    print(future_forecast_results)
