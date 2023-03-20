from pmdarima import  auto_arima
from statsmodels.tsa.stattools import adfuller
from fetch_equity import asset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd


# select asset
asset_series = asset(symbol='VUSA.AS',granularity='1d', start= '2022-01-01', end='2022-11-29')
print(asset_series)
raw = asset_series.fetch_asset()
raw.set_index('date',inplace=True)
ts = raw['return'].copy()

# stationarity -Dickey Fuller (add also kpss and decide on optimal diffs)10.1.3.1.
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
split_idx

# Split
train = ts.iloc[:split_idx]
test = ts.iloc[split_idx:]

# Visualize split
fig,ax= plt.subplots(figsize=(12,8))
plt.plot(train, label='Train', marker='o')
plt.plot(test, label='Test', marker='o')
ax.legend(bbox_to_anchor=[1,1])
plt.show()

model = auto_arima(train, start_p=0, start_q=0)
model.summary()
model.plot_diagnostics()
plt.show()

fc, conf_int = model.predict(n_periods=12, return_conf_int=True)


