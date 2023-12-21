from fetch_equity import crypto, asset
import pandas as pd
import numpy as np
import statsmodels.api as sm
import lightgbm as gbm

# select asset
asset_series = asset(symbol='IBM', granularity='1d', start='2023-01-01', end='2023-12-10')
print(asset_series)
raw = asset_series.fetch_asset()
raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
ts = raw[['date', 'typical_price']].copy()

# build covariates based on granularity / now only daily for prototype
new_cols = ['month', 'week', 'day', 'dayofweek', 'quarter']
for i in new_cols:
    ts[f'{i}'] = eval('ts["date"].dt.'+i)

ts['wom'] = ts["date"].apply(lambda d: (d.day-1) // 7 + 1)
ts.set_index('date', inplace=True)

# decide on lag number via acf
acf_values = sm.tsa.stattools.acf(ts['typical_price'], nlags=len(ts)-1)
acf_val = acf_values[1:]
significant_count = sum(acf > 0.90 for acf in acf_val)

# build lag covariates
for i in range(1,significant_count+1):
    ts[f'lag_{i}'] = ts['typical_price'].shift(i)

# keep complete cases
ts.dropna(axis='rows', inplace=True)

