from fetch_equity import crypto, asset
import pandas as pd
import numpy as np
import statsmodels.api as sm
import lightgbm as gbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from datetime import date, timedelta


# select asset
asset_series = asset(symbol='GPN', granularity='1d', start='2020-04-09', end='2024-01-05')
print(asset_series)
raw = asset_series.fetch_asset()
raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
ts = raw[['date', 'typical_price']].copy()

# build covariates based on granularity / now only daily for prototype
new_cols = ['quarter', 'month', 'week', 'day', 'dayofweek']
for i in new_cols:
    if i != 'week':
       ts[f'{i}'] = eval('ts["date"].dt.'+i)
    else: 
       ts[f'{i}'] = eval('ts["date"].dt.isocalendar().'+i)

ts['wom'] = ts["date"].apply(lambda d: (d.day-1) // 7 + 1)
ts.set_index('date', inplace=True)

# decide on lag number via acf
acf_values = sm.tsa.stattools.acf(ts['typical_price'], nlags=len(ts)-1)
acf_val = acf_values[1:]
significant_count = sum(acf > 0.99 for acf in acf_val)

# build lag covariates
for i in range(1,significant_count+1):
    ts[f'lag_{i}'] = ts['typical_price'].shift(i)

# keep complete cases
ts.dropna(axis='rows', inplace=True)

# train test split
hold_out = 0.99
if  hold_out != 0.8:
    train_size = hold_out
else:
    train_size = 0.8

split_idx = round(len(ts) * train_size)

# Split
train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]
cv_split = TimeSeriesSplit(n_splits=4, test_size=100)

# hyperparameters tuning - cv
fixed_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'bagging_freq': 5,
    'boosting_type': 'gbdt',
    'num_rounds': 100,
}

search_params = {
    'max_depth': [3, 4, 6, 5, 10],
    'num_leaves': [12, 20, 32],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'feature_fraction': [0.8, 0.9],
}

base_model = LGBMRegressor(**fixed_params)
grid_search = GridSearchCV(estimator=base_model,
                           param_grid=search_params,
                           cv=cv_split,
                           scoring="neg_root_mean_squared_error",
                           n_jobs=-1,
                           verbose=False)

X = train.loc[:, train.columns != 'typical_price']
y = train['typical_price']
grid_search.fit(X, y)
print("Best Parameters: ", grid_search.best_params_)

best_model = LGBMRegressor(**fixed_params, **grid_search.best_params_)
best_model.fit(X,y)
prediction = best_model.predict(test.loc[:, train.columns != 'typical_price'])
y_test = test['typical_price']

def evaluate_model(y_test, prediction):
  print(f"MAE: {mean_absolute_error(y_test, prediction)}")
  print(f"MSE: {mean_squared_error(y_test, prediction)}")
  print(f"MAPE: {mean_absolute_percentage_error(y_test, prediction)}")

evaluate_model(y_test,prediction)

#train model on full data
X = ts.loc[:, ts.columns != 'typical_price']
y = ts['typical_price']
best_model.fit(X, y)

#to forecast ahead - build lagged process
start_date = ts.index.max() + timedelta(days=1)
new_index_range = pd.date_range(start_date,periods=7)

column_info = [(col, str(dt)) for col, dt in zip(ts.columns, ts.dtypes)]
forecast_df = pd.DataFrame(columns=[col for col, _ in column_info])
forecast_df['typical_price']=np.repeat(np.NaN,7,axis=0)
forecast_df.set_index(new_index_range, inplace=True)
