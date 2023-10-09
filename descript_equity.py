from fetch_equity import  asset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
plt.interactive(False)
plt.style.use('ggplot')
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objects as go
import statsmodels.api as sm
import hampel as hm

# select asset
asset_series = crypto(symbol='xrp',granularity='1d', start= '2023-01-01', end='2023-06-25')
print(asset_series)
raw = asset_series.fetch_crypto()
#raw.set_index('date',inplace=True)
#raw = raw.rename(columns={'index' : 'date', 'WHEAT': 'close'})

# plot raw time series with smoothed line and bollinger band
sma = raw[['close']].rolling(14).mean()
raw['typical_price'] = (raw['high'] + raw['low'] + raw['close']) / 3
roll_std = raw[['typical_price']].rolling(14).std()
upper_boll = sma['close'] + 2*roll_std['typical_price']
lower_boll = sma['close'] - 2*roll_std['typical_price']

plt.plot(raw['close'], color = 'navy', label = 'Daily Closes')
plt.plot(sma, color = 'teal', label = 'Smoothed Line',alpha=0.8)
plt.plot(upper_boll, color = 'red', label = 'Upper Bollinger',alpha=0.5)
plt.plot(lower_boll, color = 'green', label = 'Upper Bollinger',alpha=0.5)
plt.legend(loc='upper right')
plt.xlabel('Symbol '+'Daily '+'Close')
plt.ylabel('Value')
plt.xticks(rotation = 45)
plt.show()


'''
# calculate RSI validate
raw['Up'] = np.where(raw['return']>0,raw['return'],0)
raw['Down'] = np.where(raw['return']<0,raw['return'],0)
raw['Umean'] = raw[['Up']].rolling(14).mean()
raw['Dmean'] = raw[['Down']].rolling(14).mean()
raw['RS'] = raw.apply(lambda x : x['Umean']/x['Dmean'], axis=1)
raw['RSI'] = raw.apply(lambda x : 100 - (100/1 + x['RS']), axis=1)
'''

# build covariates
cov_set = raw[['close']]
cov_set.reset_index(inplace=True)
cov_set = cov_set.rename(columns={'index' : 'date'})
new_cols = ['month','week','day','dayofweek','quarter']
for i in new_cols:
    cov_set[f'{i}'] = eval('cov_set["date"].dt.'+i)

cov_set['wom'] = cov_set["date"].apply(lambda d: (d.day-1) // 7 + 1)
cov_set.set_index('date',inplace=True)

#auto detect seasonality
#split sets
covariates = cov_set.loc[:,cov_set.columns.str.contains('close')==False].copy()
response = cov_set.loc[:,cov_set.columns.str.contains('close')==True].copy()

#build model
covariates = sm.add_constant(covariates)
model_structure = sm.OLS(response, covariates)
linear_model = model_structure.fit()
predictions = linear_model.predict(covariates)

#model summary
linear_model.summary()
# actual vs predicted plot
predictions_df = pd.Series(predictions)
predictions_df.index = response.index


# create a figure with subplots
fig, axes = plt.subplots(2, 1, figsize = (10,8))
fig.suptitle('Actual vs Predicted')
# plot the first data set in the top left subplot
sns.regplot(ax = axes[0],x=response, y=predictions)
# plot the second data set in the top right subplot
axes[1].plot(response, color = 'navy', label = 'Actual')
axes[1].plot(predictions_df, color = 'teal', label = 'Predicted',alpha=0.7)
plt.xticks(rotation = 45)
plt.show()

# plot seasonality and trend plots  # double or triple based on pvals based on mstl - to be added
decompose_result_mult = seasonal_decompose(raw[['close']], period=90)

fig, axs = plt.subplots(ncols=2, nrows=2, figsize = (10,8))
ax1, ax2, ax3, ax4 = axs.flat
fig.suptitle('Seasonality & Trend')

ax1.plot(decompose_result_mult.seasonal,color = 'navy')
ax1.set_ylabel('Seasonal Index')
ax1.tick_params(labelrotation=45)

ax2.plot(decompose_result_mult.trend, color = 'navy')
ax2.set_xlabel('Date')
ax2.set_ylabel('Trend line')
ax2.tick_params(labelrotation=45)

ax3.plot(decompose_result_mult.resid, color = 'navy')
ax3.set_xlabel('Date')
ax3.set_ylabel('Residuals')
ax3.tick_params(labelrotation=45)

ax4.plot(decompose_result_mult.observed, color = 'navy')
ax4.set_xlabel('Date')
ax4.set_ylabel('Observed')
ax4.tick_params(labelrotation=45)

plt.show()

#decompose_result_mult.plot().show()
# plot candle stick

# plt.figure()
# up = raw[raw.close >= raw.open]
# down = raw[raw.close < raw.open]
# #plot parameters
# col1,col2,width,width2 = 'navy','teal',5,.5
#
# plt.bar(up.index, up.close-up.open, width, bottom= up.open, color = col2)
# plt.bar(up.index, up.high-up.close, width2, bottom=up.close, color=col1)
# plt.bar(up.index, up.low-up.open, width2, bottom=up.open, color=col1)
#
# plt.bar(down.index, down.close-down.open, width, bottom=down.open, color=col2)
# plt.bar(down.index, down.high-down.open, width2, bottom=down.open, color=col2)
# plt.bar(down.index, down.low-down.close, width2, bottom=down.close, color=col2)
# plt.xticks(rotation=30, ha='right')
# plt.show()
fig = go.Figure(data=[go.Candlestick(x=raw.index,
                open=raw.open,
                high=raw.high,
                low=raw.low,
                close=raw.close)])

fig.show()
# plot acf pacf
# create a figure with subplots
fig, axes = plt.subplots(2, 1, figsize = (10,8))
# plot the first data set in the top left subplot
plot_acf(raw[['close']],ax = axes[0] ,lags=20)
# plot the second data set in the top right subplot
plot_pacf(raw[['close']],ax = axes[1] ,lags=20)
plt.show()


# outlier analysis -- hampel filter -- MAD
outlier_indices = hm.hampel(raw['close'],
                            window_size= 7,
                            n=2,
                            imputation=False)


plt.plot(raw['close'], color = 'navy', label = 'Daily Closes')
plt.scatter(raw.iloc[outlier_indices].index,raw.iloc[outlier_indices]['close'], color = 'orange', s = 50)
plt.xlabel('Symbol '+'Daily '+'Close')
plt.ylabel('Value')
plt.xticks(rotation = 45)
plt.show()




