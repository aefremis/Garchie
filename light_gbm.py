from fetch_equity import crypto, asset
import pandas as pd
import numpy as np
import statsmodels.api as sm
import lightgbm as gbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV


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

# train test split
hold_out = 0.95
if  hold_out != 0.8:
    train_size = hold_out
else:
    train_size = 0.8

split_idx = round(len(ts) * train_size)

# Split
train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

# hyperparameters tuning - cv
fixed_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'bagging_freq': 5,
    'boosting_type': 'gbdt',
    'num_rounds': 100,
}

search_params = {
    'num_leaves': [12, 20, 32],
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.7, 0.8, 0.9],
}

base_model = LGBMRegressor(**fixed_params)
grid_search = GridSearchCV(estimator=base_model,
                           param_grid=search_params,
                           cv=5,
                           scoring="neg_root_mean_squared_error",
                           n_jobs=-1,
                           verbose=False)

X = train.loc[:, train.columns != 'typical_price']
y = train['typical_price']
grid_search.fit(X, y)
print("Best Parameters: ", grid_search.best_params_)

best_model = LGBMRegressor(**fixed_params, **grid_search.best_params_)
best_model.fit(X,y)
best_model.predict(test.loc[:, train.columns != 'typical_price'])
test['typical_price']

#fix it