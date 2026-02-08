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
    
    def design_gam_model(self) -> pd.DataFrame:
        """
        Fits the GAM model using Prophet and generates forecasts.

        Returns
        -------
        pd.DataFrame
            Standardized forecast DataFrame with columns:
            ['date', 'prediction', 'model_name', 'variable', 'lower_bound', 'upper_bound']
        """

         # make prophet dataset
        gam_df = self.ts.rename(columns={'date':'ds','typical_price':'y'})
        model = ph.Prophet()
        model.fit(gam_df)

        freq = self._get_freq()

        future = model.make_future_dataframe(periods = self.forecast_ahead, freq=freq)
        forecast = model.predict(future)
        
        # Standardize return structure
        pred_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(self.forecast_ahead).copy()
        pred_df.rename(columns={
            'ds': 'date',
            'yhat': 'prediction',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        }, inplace=True)
        
        pred_df['model_name'] = 'GAM'
        pred_df['variable'] = 'price'
        
        # Reorder columns
        pred_df = pred_df[['date', 'prediction', 'model_name', 'variable', 'lower_bound', 'upper_bound']]
        
        # diagnostics
        if self.diagnostics:
            EDA.plot_gam_diagnostics(model, forecast)

        return pred_df
    

if __name__ == "__main__":
    # select asset
    asset_series = asset(symbol='MSFT', granularity='1wk', start='2023-01-01', end='2025-12-29')
    print(asset_series)
    raw = asset_series.fetch_asset()
    raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3, 2)
    ts = raw[['date', 'typical_price']].copy()

    gm = gam_model(ts=ts, forecast_ahead=25, forecast_unit='weeks', diagnostics=True)
    print(gm)
    forecast_results = gm.design_gam_model()
    print(forecast_results)