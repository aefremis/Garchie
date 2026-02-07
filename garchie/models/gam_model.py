from garchie.data import crypto, asset
import pandas as pd
import numpy as np
import prophet as ph

if __name__ == "__main__":
    # select asset
    asset_series = asset(symbol='MSFT', granularity='1wk', start='2023-01-01', end='2025-12-29')
    print(asset_series)
    raw = asset_series.fetch_asset()
    raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
    ts = raw[['date', 'typical_price']].copy()


class gam_model:
    """
    A class used to represent the Generalized Additive Model (GAM) using Prophet for time series forecasting.

    ...
    Attributes
    ----------
    ts : dataframe
        a dataframe that contains the asset information (dates and prices)
    forecast_ahead : integer
        a number that dictates the forecasting horizon
    forecast_unit : string
        the unit of time for forecasting ('days', 'weeks', 'months')
    diagnostics : bool
        a boolean that enables diagnostic plotting

    Methods
    -------
    design_gam_model()
        Designs the GAM model using Prophet and generates forecasts
    """

    def __init__(self, ts, forecast_ahead, forecast_unit, diagnostics=True):
        """
        Parameters
        ----------
        ts : dataframe
            a dataframe that contains the asset information (dates and prices)
        forecast_ahead : integer
            a number that dictates the forecasting horizon
        forecast_unit : string
            the unit of time for forecasting ('days', 'weeks', 'months')
        diagnostics : bool
            a boolean that enables diagnostic plotting
        """
        self.ts = ts
        self.forecast_ahead = forecast_ahead
        self.forecast_unit = forecast_unit.lower()
        self.diagnostics = diagnostics 

    def __str__(self):
        return f"class of gam model"
    

       
    def _get_freq(self):
        """
        Maps the forecast unit to a pandas frequency string.

        Returns
        -------
        string :
            pandas frequency alias ('D', 'W', 'MS')
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
    
    def design_gam_model(self):
        """
        Designs the GAM model using Prophet and generates forecasts.

        Parameters
        ----------
        self : an object of class gam_model

        Returns
        -------
        DataFrame :
            a standardized dataframe containing the forecasted values with columns:
            date, prediction, model_name, variable, lower_bound, upper_bound
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
        
        print(pred_df)

         # diagnostics
        if self.diagnostics:
            fig = model.plot_components(forecast)
            from prophet.plot import plot_plotly, plot_components_plotly
            plot_plotly(model, forecast).show()
            plot_components_plotly(model, forecast).show()

        return(pred_df)
    

'''
gm = gam_model(ts=ts, forecast_ahead=25, forecast_unit='weeks', diagnostics=True)
print(gm)
gm.design_gam_model()
'''