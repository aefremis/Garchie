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
from sklearn.linear_model import LinearRegression
from scipy import stats

# select asset
asset_series = crypto(symbol='btc',granularity='1d', start= '2022-01-01', end='2022-11-29')
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
lin_mod = LinearRegression(fit_intercept=True)
#split sets
covariates = cov_set.loc[:,cov_set.columns.str.contains('close')==False].copy()
response = cov_set.loc[:,cov_set.columns.str.contains('close')==True].copy()
#fit & predict model
lin_mod.fit(covariates,response)
params =np.append(lin_mod.intercept_,lin_mod.coef_)
predictions = lin_mod.predict(covariates)

# calculate p-values
new_X = np.append(np.ones((len(covariates),1)),
                  covariates,
                  axis=1)
MSE = (sum((response.to_numpy() - predictions)**2))/(len(new_X)-len(new_X[0]))
critical_scores = params/ np.sqrt(MSE*(np.linalg.inv(np.dot(new_X.T,
                                                            new_X)).diagonal()))

# two sided p values
p_val =[2*(1-stats.t.cdf(np.abs(i),(len(new_X)-len(new_X[0])))) for i in critical_scores]
p_val = np.round(p_val,4)
coefs = np.append(lin_mod.intercept_,lin_mod.coef_).flatten()
covariates_names = covariates.columns.insert(0,"Intercept")

# significance dictionary
sig_dic = {'variables':covariates_names,
           'coef':coefs,
           'p_vals':p_val}
sig_df = pd.DataFrame.from_dict(sig_dic)

# Coefficient of determination
R_sq = round(lin_mod.score(covariates,
                           response),2)

# regression plot
sns.regplot(x=response, y=predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# actual vs predicted plot
predictions_df = pd.Series(predictions.flatten())
predictions_df.index = response.index

plt.plot(response, color = 'navy', label = 'Actual')
plt.plot(predictions_df, color = 'teal', label = 'Predicted',alpha=0.7)
plt.legend(loc='upper right')
plt.ylabel('Value')
plt.xlabel('Date')
plt.title("Actual vs Predicted Values")
plt.xticks(rotation = 45)
plt.show()

# # create a figure with subplots
# fig, axs = plt.subplots(nrows= 2, ncols= 2)
#
# # plot the first data set in the top left subplot
# sns.regplot(ax = axs[0, 0],x=response, y=predictions)
#
# # plot the second data set in the top right subplot
# axs[0, 1].plot(response, color = 'navy', label = 'Actual')
# axs[0, 1].plot(predictions_df, color = 'teal', label = 'Predicted',alpha=0.7)
#
# plt.show()

# plot seasonality and trend plots  # double or triple based on pvals based on mstl - to be added
decompose_result_mult = seasonal_decompose(raw[['close']], period=30)

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

