# libs and data
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import arch
from fetch_equity import crypto, asset

# select asset
asset_series = asset(symbol='VUSA.AS',granularity='1d', start= '2019-01-01', end='2023-06-25')
print(asset_series)
raw = asset_series.fetch_asset()

#raw.set_index('date',inplace=True)
ts = raw['return'].copy()


# Calculate daily std of returns
'''
Astatheia
Volatility is typically unobservable,
and as such estimated --- for example via the (sample) variance of returns, or more frequently,
its square root yielding the standard deviation of returns as a volatility estimate.
There are also countless models for volatility,
    from old applied models like Garman/Klass to exponential
    decaying and formal models such as GARCH or Stochastic Volatility.
As for forecasts of the movement: well, that is a different topic as movement is 
the first moment (mean, location) whereas volatility is a second moment (dispersion, variance, volatility). 
So in a certain sense, volatility estimates do not give you estimates of future direction but of future ranges of movement.
'''

def calc_volatility(period, returns_ts):
    # calculate volatility as the standard deviation of variance
    mean_returns = round(returns_ts.abs().mean(),2)
    std_returns = returns_ts.std()
    # turn into period volatility
    period_volatility = round(np.sqrt(period) * std_returns, 2)
    print('Single steps absolute changes: ', '{:.2f}%'.format(mean_returns))
    print('Period volatility: ', '{:.2f}%'.format(period_volatility))
    return mean_returns, period_volatility

calc_volatility(14,ts)

## Implement a basic GARCH model

# Specify GARCH model assumptions
basic_gm = arch.arch_model(ts, p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')
# Fit the model
gm_result = basic_gm.fit(update_freq = 4)

## Plot distribution of standardized residuals

# Obtain model estimated residuals and volatility
gm_resid = gm_result.resid
gm_std = gm_result.conditional_volatility

# Calculate the standardized residuals
gm_std_resid = gm_resid /gm_std

# Plot the histogram of the standardized residuals
plt.hist(gm_std_resid, bins = 50, 
         facecolor = 'orange', label = 'Standardized residuals')
plt.legend(loc = 'upper left')
plt.show()

## Fit a GARCH with skewed t-distribution
# Specify GARCH model assumptions
skewt_gm = arch.arch_model(raw['return'], p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'skewt')
# Fit the model
skewt_result = skewt_gm.fit()
cmean_vol = skewt_result.conditional_volatility
# Specify GARCH model assumptions
armean = arch.arch_model(raw['return'], p = 1, q = 1, mean = 'AR', vol = 'GARCH', dist = 'skewt')
# Fit the model
armean_result = armean.fit()
armean_vol = armean_result.conditional_volatility

## Effect of mean model on volatility predictions

# Print model summary of GARCH with constant mean
print(skewt_result.summary())
# Print model summary of GARCH with AR mean
print(armean_result.summary())

# Plot model volatility 
plt.plot(cmean_vol, color = 'blue', label = 'Constant Mean Volatility')
plt.plot(armean_vol, color = 'red', label = 'AR Mean Volatility')
plt.legend(loc = 'upper right')
plt.show()

# Check correlation of volatility estimations
print(np.corrcoef(cmean_vol, armean_vol)[0,1])

##Fit GJR- GARCH models to ts

# Specify model assumptions
gjr_gm = arch.arch_model(ts, p = 1, q = 1, o = 1, vol = 'GARCH', dist = 't')

# Fit the model
gjrgm_result = gjr_gm.fit(disp = 'off')
gjrgm_vol = gjrgm_result.conditional_volatility
# Print model fitting summary
print(gjrgm_result.summary())

##Fit E- GARCH models to ts

# Specify model assumptions
e_gm = arch.arch_model(ts, p = 1, q = 1, o = 1, vol = 'EGARCH', dist = 't')

# Fit the model
e_result = e_gm.fit(disp = 'off')
e_vol = e_result.conditional_volatility
# Print model fitting summary
print(e_result.summary())

## Compare GJR-GARCH with EGARCH

# Plot the actual Bitcoin returns
plt.plot(ts, color = 'grey', alpha = 0.4, label = 'Price Returns')

# Plot GJR-GARCH estimated volatility
plt.plot(gjrgm_vol, color = 'gold', label = 'GJR-GARCH Volatility')

# Plot EGARCH  estimated volatility
plt.plot(e_vol, color = 'red', label = 'EGARCH Volatility')

plt.legend(loc = 'upper right')
plt.show()
# Print each models BIC
print(f'GJR-GARCH BIC: {gjrgm_result.bic}')
print(f'\nEGARCH BIC: {e_result.bic}')


## Fixed rolling window forecast
ts.index = raw.date
index = ts.index
start_loc = 0
end_loc = np.where(index >= '2022-11-24')[0].min()
forecasts = {}
from arch.__future__ import reindexing
for i in range(len(ts) - end_loc):
    sys.stdout.write('o')
    sys.stdout.flush()
    res = gjr_gm.fit(first_obs=start_loc + i, last_obs=i + end_loc, disp='off')
    temp = res.forecast(horizon=1,start=end_loc).variance
    fcast = temp.iloc[i]
    forecasts[fcast.name] = fcast
print(' Done!')
variance_fixedwin = pd.DataFrame(forecasts).T


## Expanding window forecast
ts.index = raw.date
index = ts.index
start_loc = 0
end_loc = np.where(index >= '2022-11-24')[0].min()
forecasts = {}
for i in range(len(ts) - end_loc):
    sys.stdout.write('-')
    sys.stdout.flush()
    res = gjr_gm.fit(first_obs = start_loc, last_obs = i + end_loc, disp = 'off')
    temp = res.forecast(horizon=1,start=end_loc).variance
    fcast = temp.iloc[i]
    forecasts[fcast.name] = fcast
print(' Done!')
variance_expandwin = pd.DataFrame(forecasts).T
## Compare forecast results

# Print top 5 rows of variance forecast with an expanding window
print(variance_expandwin.head(5))
# Print top 5 rows of variance forecast with a fixed rolling window
print(variance_fixedwin.head(5))

##Simplify the model with p-values

# Print model fitting summary
print(gjrgm_result.summary())

# Get parameter stats from model summary
para_summary = pd.DataFrame({'parameter':gjrgm_result.params,
                             'p-value': gjrgm_result.pvalues})

# Print out parameter stats
print(para_summary)

## Simplify the model with t-statistics

# Get parameter stats from model summary
para_summary = pd.DataFrame({'parameter':gjrgm_result.params,
                             'std-err': gjrgm_result.std_err,
                             't-value': gjrgm_result.tvalues})
# Print parameter stats
print(para_summary)

## ACF plot
# Import the Python module
from statsmodels.graphics.tsaplots import plot_acf

# Plot the standardized residuals
plt.plot(gjrgm_result.resid.dropna())
plt.title('Standardized Residuals')
plt.show()

# Generate ACF plot of the standardized residuals
plot_acf(gjrgm_result.resid.dropna(), alpha = 0.05)
plt.show()

## Ljung-Box test

# Import the Python module
from statsmodels.stats.diagnostic import acorr_ljungbox

# Perform the Ljung-Box test
lb_test = acorr_ljungbox(gjrgm_result.std_resid , lags = 10)

# Store p-values in DataFrame
df = pd.DataFrame({'P-values': lb_test['lb_pvalue']}).T

# Create column names for each lag
col_num = df.shape[1]
col_names = ['lag_'+str(num) for num in list(range(1,col_num+1,1))]

# Display the p-values
df.columns = col_names
df

# Display the significant lags
mask = df < 0.05
df[mask].dropna(axis=1)

# Print the log-likelihodd of normal GARCH
## In general, the bigger the log-likelihood, the better the model since it implies a bigger probability of having observed the data you got.
print('Log-likelihood of normal GJR-GARCH :', gjrgm_result.loglikelihood)
# Print the log-likelihodd of skewt GARCH
print('Log-likelihood of skewt GARCH :', e_result.loglikelihood)


## Pick a winner based on AIC/BIC
## The lower the AIC or BIC, the better the model.
# Print the AIC GJR-GARCH
print('AIC of GJR-GARCH model :', gjrgm_result.aic)
# Print the AIC of EGARCH
print('AIC of EGARCH model :', e_result.aic)

# Print the BIC GJR-GARCH
print('BIC of GJR-GARCH model :', gjrgm_result.bic)
# Print the BIC of EGARCH
print('BIC of EGARCH model :', e_result.bic)

## Backtesting with MAE, MSE
from sklearn.metrics import mean_absolute_error, mean_squared_error
def evaluate(observation, forecast): 
    # Call sklearn function to calculate MAE
    mae = mean_absolute_error(observation, forecast)
    print('Mean Absolute Error (MAE): {:.3g}'.format(mae))
    # Call sklearn function to calculate MSE
    mse = mean_squared_error(observation, forecast)
    print('Mean Squared Error (MSE): {:.3g}'.format(mse))
    return mae, mse

# Backtest model with MAE, MSE
evaluate(raw['return'].sub(raw['return'].mean()).pow(2), gjrgm_vol**2)
evaluate(raw['return'].sub(raw['return'].mean()).pow(2), e_vol**2)

# Plot the actual  volatility
plt.plot(raw['return'].sub(raw['return'].mean()).pow(2),
         color = 'grey', alpha = 0.4, label = 'Daily Volatility')

# Plot EGARCH  estimated volatility
plt.plot(gjrgm_vol**2, color = 'red', label = 'GJR-GARCH Volatility')

plt.legend(loc = 'upper right')
plt.show()

### connect predictions with returns