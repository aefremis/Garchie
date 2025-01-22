from fetch_equity import crypto, asset
import pandas as pd
import numpy as np
import prophet as ph
# select asset
asset_series = asset(symbol='MSFT', granularity='1d', start='2023-01-01', end='2023-12-29')
print(asset_series)
raw = asset_series.fetch_asset()
raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
ts = raw[['date', 'typical_price']].copy()


class gam_model:

    def __init__(self, ts, forecast_ahead, diagnostics):
        self.ts = ts
        self.forecast_ahead = forecast_ahead
        self.diagnostics = diagnostics 

    def __str__(self):
        return f"class of gam model"
    
    def design_gam_model(self):

         # make prophet dataset
        gam_df = self.ts.rename(columns={'date':'ds','typical_price':'y'})
        model = ph.Prophet()
        model.fit(gam_df)

        future = model.make_future_dataframe(periods = self.forecast_ahead)
        forecast = model.predict(future)
        pred_df = forecast[['ds', 'yhat']].tail(7)

        if self.diagnostics:
            fig = model.plot_components(forecast)
            from prophet.plot import plot_plotly, plot_components_plotly, pl
            plot_plotly(model, forecast).show()

        return(pred_df)
    


gm = gam_model(ts=ts, forecast_ahead=7, diagnostics=False)
print(gm)
gm.design_gam_model()