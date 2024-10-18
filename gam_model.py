from fetch_equity import crypto, asset
import pandas as pd
import numpy as np
import prophet as ph
# select asset
asset_series = asset(symbol='GPN', granularity='1d', start='2023-01-01', end='2024-07-11')
print(asset_series)
raw = asset_series.fetch_asset()
raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
ts = raw[['date', 'typical_price']].copy()

# make prophet dataset
gam_df = ts.rename(columns={'date':'ds','typical_price':'y'})

model = ph.Prophet()
model.fit(gam_df)

future = model.make_future_dataframe(periods = 14)

forecast = model.predict(future)

forecast[['ds', 'yhat']].tail(14)

fig = model.plot_components(forecast)
from prophet.plot import plot_plotly, plot_components_plotly, pl

plot_plotly(model, forecast).show()