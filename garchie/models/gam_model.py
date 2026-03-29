from garchie.data import crypto, asset, commodity
import pandas as pd
import numpy as np
import prophet as ph
from garchie.eda import EDA

class gam_model:
    """
    Implements a Generalized Additive Model (GAM) using Prophet for time series forecasting.

    Attributes
    ----------
    ts : pd.DataFrame
        Input DataFrame containing 'date' and 'typical_price' columns.
    forecast_ahead : int
        Number of steps to forecast into the future.
    forecast_unit : str
        Time unit for forecasting ('days', 'weeks', 'months').
    diagnostics : bool
        If True, displays diagnostic plots during execution.

    Methods
    -------
    design_gam_model()
        Fits the GAM model using Prophet and generates forecasts.
    """

    def __init__(self, ts: pd.DataFrame, forecast_ahead: int, forecast_unit: str, diagnostics: bool = True):
        """
        Parameters
        ----------
        ts : pd.DataFrame
            Input DataFrame containing 'date' and 'typical_price' columns.
        forecast_ahead : int
            Number of steps to forecast into the future.
        forecast_unit : str
            Time unit for forecasting ('days', 'weeks', 'months').
        diagnostics : bool
            If True, displays diagnostic plots during execution.
        """
        self.ts = ts
        self.forecast_ahead = forecast_ahead
        self.forecast_unit = forecast_unit.lower()
        self.diagnostics = diagnostics 

    def __str__(self):
        return f"GAM Model (Forecast: {self.forecast_ahead} {self.forecast_unit})"
    

       
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
    
    def design_gam_model(self, validation_steps: int) -> pd.DataFrame:
        """
        Fits the GAM model using Prophet and generates both validation and future forecasts.

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
        # --- Validation Predictions ---
        # Split data for validation
        ts_indexed = self.ts.copy()
        ts_indexed['date'] = pd.to_datetime(ts_indexed['date']) # Ensure date is datetime
        
        train_validation_df = ts_indexed.iloc[:-validation_steps].rename(columns={'date': 'ds', 'typical_price': 'y'})
        test_validation_dates = ts_indexed.iloc[-validation_steps:]['date']

        # Fit model on training data for validation
        model_validation = ph.Prophet()
        model_validation.fit(train_validation_df)

        freq = self._get_freq()

        # Generate future dataframe for validation period
        future_validation = model_validation.make_future_dataframe(periods=validation_steps, freq=freq, include_history=False)
        
        # Predict validation period
        forecast_validation = model_validation.predict(future_validation)
        
        # Standardize validation return structure
        validation_pred_df = forecast_validation[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        validation_pred_df.rename(columns={
            'ds': 'date',
            'yhat': 'predicted_values',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        }, inplace=True)
        
        validation_pred_df['model_name'] = 'GAM'
        validation_pred_df['variable'] = 'price'
        validation_pred_df['type'] = 'validation'
        
        # Reorder columns
        validation_pred_df = validation_pred_df[['date', 'predicted_values', 'lower_bound', 'upper_bound', 'model_name', 'variable', 'type']]

        # --- Future Forecast Predictions ---
        # Make prophet dataset from full data
        gam_df_full = self.ts.rename(columns={'date':'ds','typical_price':'y'})
        model_full = ph.Prophet()
        model_full.fit(gam_df_full)

        # Generate future dataframe for actual forecast_ahead
        future_forecast = model_full.make_future_dataframe(periods=self.forecast_ahead, freq=freq, include_history=False)
        
        # Predict future period
        forecast_full = model_full.predict(future_forecast)
        
        # Standardize future forecast return structure
        future_pred_df = forecast_full[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        future_pred_df.rename(columns={
            'ds': 'date',
            'yhat': 'predicted_values',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        }, inplace=True)
        
        future_pred_df['model_name'] = 'GAM'
        future_pred_df['variable'] = 'price'
        future_pred_df['type'] = 'future'
        
        # Reorder columns
        future_pred_df = future_pred_df[['date', 'predicted_values', 'lower_bound', 'upper_bound', 'model_name', 'variable', 'type']]
        
        # diagnostics for full model (optional, can be adapted)
        if self.diagnostics:
            EDA.plot_gam_diagnostics(model_full, forecast_full)

        return pd.concat([validation_pred_df, future_pred_df], ignore_index=True)
    

if __name__ == "__main__":
    # select asset
    asset_series = asset(symbol='MSFT', granularity='1wk', start='2023-01-01', end='2025-12-29')
    print(asset_series)
    raw = asset_series.fetch_asset()
    raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3, 2)
    ts = raw[['date', 'typical_price']].copy()

    gm = gam_model(ts=ts, forecast_ahead=25, forecast_unit='weeks', diagnostics=True)
    print(gm)
    
    results_df = gm.design_gam_model(validation_steps=10) # Added validation_steps
    print(results_df)

    validation_results = results_df[results_df['type'] == 'validation']
    future_forecast_results = results_df[results_df['type'] == 'future']

    print("\nValidation Predictions:")
    print(validation_results)
    print("\nFuture Forecast:")
    print(future_forecast_results)