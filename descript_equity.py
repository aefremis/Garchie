from fetch_equity import asset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.interactive(False)
plt.style.use('ggplot')
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# select asset
asset_series = asset(symbol='VUSA.AS',
                     granularity='d',
                     start= '2020-06-01',
                     end='2022-12-07')
print(asset_series)
raw = asset_series.fetch_asset()
raw.set_index('date',inplace=True)

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


# plot seasonality and trend plots
#auto detect seasonality based on covariates lm

decompose_result_mult = seasonal_decompose(raw[['close']], period=62)

fig, axs = plt.subplots(ncols=2, nrows=2)
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


# plot acf pacf
# outlier analysis

# plot box plots per category







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

