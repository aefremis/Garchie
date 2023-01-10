from fetch_equity import crypto
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

# select asset
asset_series = crypto(symbol='btc',granularity='1d', start= '2022-09-01', end='2022-11-29')
print(asset_series)
raw = asset_series.fetch_crypto()
#raw.set_index('date',inplace=True)

# plot raw time series with smoothed line
sma = raw[['close']].rolling(14).mean()
plt.plot(raw['close'], color = 'navy', label = 'Daily Closes')
plt.plot(sma, color = 'teal', label = 'Smoothed Line',alpha=0.8)
plt.legend(loc='upper right')
plt.xlabel('Symbol '+'Daily '+'Close')
plt.ylabel('Value')
plt.xticks(rotation = 45)
plt.show()

# build covariates
cov_set = raw[['close']]
cov_set.reset_index(inplace=True)

new_cols = ['month','week','day','dayofweek','quarter']
for i in new_cols:
    cov_set[f'{i}'] = eval('cov_set["date"].dt.'+i)

cov_set['wom'] = cov_set["date"].apply(lambda d: (d.day-1) // 7 + 1)
cov_set.set_index('date',inplace=True)

#auto detect seasonality
#linear model
lin_mod = LinearRegression()
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
decompose_result_mult = seasonal_decompose(raw[['close']], period=7)

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
plot_acf(raw[['close']], lags=30)
plot_pacf(raw[['close']], lags=30)

plt.show()


# create a figure with subplots
fig, axes = plt.subplots(2, 1, figsize = (10,8))
# plot the first data set in the top left subplot
plot_acf(raw[['close']],ax = axes[0] ,lags=30)
# plot the second data set in the top right subplot
plot_pacf(raw[['close']],ax = axes[1] ,lags=30)
plt.show()


# outlier analysis








# plot the data

# calculate volatility as the standard deviation of variance
mean_daily = crypto['Return'].abs().mean()
print('Mean Daily absolute changes: ', '{:.2f}%'.format(mean_daily))
std_daily = crypto['Return'].std()
print('Daily volatility: ', '{:.2f}%'.format(std_daily))

# turn into monthly volatility
# x =  21 trading days
x = 21
month_volatility = round(np.sqrt(x) * std_daily,
                         2)
print('Monthly volatility: ', '{:.2f}%'.format(month_volatility))

