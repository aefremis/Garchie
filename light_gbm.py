from fetch_equity import crypto, asset
import pandas as pd
import numpy as np
import statsmodels.api as sm
import lightgbm as gbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from datetime import date, timedelta
import matplotlib
import matplotlib.pyplot as plt 


# select asset
asset_series = asset(symbol='GPN', granularity='1d', start='2020-04-09', end='2024-01-05')
print(asset_series)
raw = asset_series.fetch_asset()
raw['typical_price'] = np.round((raw['high'] + raw['low'] + raw['close']) / 3,2)
ts = raw[['date', 'typical_price']].copy()

# build covariates based on granularity / now only daily for prototype
def design_date_covariates(data):
    new_cols = ['quarter', 'month', 'week', 'day', 'dayofweek']
    for i in new_cols:
        if i != 'week':
            data[f'{i}'] = eval('data["date"].dt.'+i)
        else: 
            data[f'{i}'] = eval('data["date"].dt.isocalendar().'+i)

    data['wom'] = data["date"].apply(lambda d: (d.day-1) // 7 + 1)
    data.set_index('date', inplace=True)

design_date_covariates(ts)   
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
hold_out = 0.9
if  hold_out != 0.8:
    train_size = hold_out
else:
    train_size = 0.8

split_idx = round(len(ts) * train_size)

# Split
train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]
cv_split = TimeSeriesSplit(n_splits=4, test_size=100)

# visualize split
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(train['typical_price'], label='Train', marker='.')
plt.plot(test['typical_price'], label='Test', marker='.')
ax.legend(bbox_to_anchor=[1, 1])
plt.show()


# hyperparameters tuning - cv
fixed_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'bagging_freq': 5,
    'boosting_type': 'gbdt',
    'num_rounds': 100,
}

search_params = {
    'max_depth': [3, 4], #, 6, 5, 10
    'num_leaves': [12], #,20, 32
    'learning_rate': [0.1], #0.01, 0.05,
    'n_estimators': [100], # 200
    'feature_fraction': [0.9], #0.8
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

# visualize model predicting test data

pred_df = pd.DataFrame({'pred': prediction})
pred_df.set_index(test.index, inplace=True)
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(train['typical_price'], label='Train', marker='.')
ax.plot(test['typical_price'], label='Test', marker='.')
ax.plot(pred_df['pred'], label='Prediction', ls='--', linewidth=1)
ax.fill_between(x=pred_df.index, y1=pred_df['lower'], y2=pred_df['upper'], alpha=0.3)
ax.set_title('Model Validation', fontsize=22)
ax.legend(loc='upper left')
fig.tight_layout()
plt.show()

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

forecast_df.reset_index(inplace=True)
forecast_df.rename(columns={'index':'date'},
                   inplace=True)

design_date_covariates(forecast_df)

if significant_count > 0:
    lag_sequence = np.arange(1, significant_count + 1)
    forecast_df.loc[forecast_df.index[0], [f'lag_{i}' for i in lag_sequence]] = y.tail(significant_count).values

    # add lags to forecast df
    for pred_step in np.arange(7):
        temp_newdata = forecast_df.loc[[forecast_df.index[pred_step]], ~forecast_df.columns.isin(['typical_price'])]
        temp_newdata[[f'lag_{i}' for i in lag_sequence]] = temp_newdata[[f'lag_{i}' for i in lag_sequence]].astype(float)
        # temp_newdata.dtypes
        print(pred_step)

        # prediction step for one step ahead
        prediction_step = best_model.predict(temp_newdata)
        forecast_df.at[forecast_df.index[pred_step], 'typical_price'] = prediction_step[0].round(2)
        forecast_df.at[forecast_df.index[pred_step + 1], 'lag_1'] = prediction_step[0].round(2)

        if len(lag_sequence) > 1:
            for lag_step in lag_sequence[:-1]:
                forecast_df.at[forecast_df.index[pred_step + 1], f'lag_{lag_step + 1}'] = forecast_df.at[forecast_df.index[pred_step], f'lag_{lag_step}']
        else:
            print("ok")

else:
    temp_newdata = forecast_df.loc[:, ~forecast_df.columns.isin(['typical_price'])]
    forecast_df['typical_price'] = best_model.predict(temp_newdata)

# fix warning in step forecast

# visualize forecasts
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(train['typical_price'], label='Train', marker='.')
ax.plot(test['typical_price'], label='Test', marker='.')
ax.plot(pred_df['pred'], label='Prediction', ls='--', linewidth=2)
ax.plot(forecast_df['typical_price'], label='Prediction', marker = '.')
ax.set_title('Model Predictions', fontsize=22)
ax.legend(loc='upper left')
fig.tight_layout()
plt.show()
