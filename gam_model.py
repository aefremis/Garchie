from fetch_equity import crypto, asset
import pandas as pd
import numpy as np
import prophet as ph
# select asset
asset_series = asset(symbol='GPN', granularity='1d', start='2021-01-01', end='2024-07-11')
print(asset_series)
raw = asset_series.fetch_asset()
raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
ts = raw[['date', 'typical_price']].copy()

