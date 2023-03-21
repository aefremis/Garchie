from pmdarima import auto_arima, arima
from pmdarima.arima import ndiffs
from statsmodels.tsa.stattools import adfuller
from fetch_equity import crypto
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

# select asset
asset_series = crypto(symbol='btc',granularity='1d', start= '2022-01-01', end='2022-11-29')
print(asset_series)
raw = asset_series.fetch_crypto()
#raw.set_index('date',inplace=True)
ts = raw['return'].copy()

# stationarity -Dickey Fuller (add also kpss and decide on optimal diffs)
# p value < 0.05 -- stationary time series
dftest = adfuller(ts)
print('\nResults of Dickey-Fuller Test: \n')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value',
                                         '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)


# Train Test Split Index
train_size = 0.8
split_idx = round(len(ts)* train_size)

# Split
train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]


# Visualize split
fig,ax= plt.subplots(figsize=(12,8))
plt.plot(train, label='Train', marker='o')
plt.plot(test, label='Test', marker='o')
ax.legend(bbox_to_anchor=[1,1])
plt.show()

# d term
kpss_diffs = ndiffs(train, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(train, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)
print(f"Estimated differencing term: {n_diffs}")


model = auto_arima(train, d=n_diffs, start_p=0, start_q=0)
model.summary()
model.plot_diagnostics()
plt.show()

fc, conf_int = model.predict(n_periods=66, return_conf_int=True)
pred_df = pd.DataFrame({'pred' : fc,
                        'lower' : conf_int[:,0],
                        'upper' : conf_int[:,1]})
pred_df.set_index(test.index,inplace=True)

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(train, label='Train', marker='o')
ax.plot(test, label='Test', marker='o')
ax.plot(pred_df['pred'], label='Prediction', ls='--', linewidth=3)

ax.fill_between(x=pred_df.index, y1=pred_df['lower'], y2=pred_df['upper'], alpha=0.3)
ax.set_title('Model Validation', fontsize=22)
ax.legend(loc='upper left')
fig.tight_layout()
plt.show()

# predict future
best_model = SARIMAX(ts,
                     order=model.order,
                     seasonal_order=model.seasonal_order).fit()
best_model.summary()


forecast = best_model.get_forecast(steps=15)
pred_df = forecast.conf_int()
pred_df['pred'] = forecast.predicted_mean
pred_df.columns = ['lower', 'upper', 'pred']

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(train, label='Train', marker='o')
ax.plot(test, label='Test', marker='o')
ax.plot(pred_df['pred'], label='Prediction', ls='--', linewidth=3)

ax.fill_between(x=pred_df.index, y1=pred_df['lower'], y2=pred_df['upper'], alpha=0.3)
ax.set_title('Model Validation', fontsize=22)
ax.legend(loc='upper left')
fig.tight_layout()
plt.show()