# libs and data
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import arch
from fetch_equity import crypto, asset
from itertools import product
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from arch.__future__ import reindexing


# select asset
asset_series = asset(symbol='vusa.as',granularity='1d', start= '2023-01-01', end='2023-11-21')
print(asset_series)
raw = asset_series.fetch_asset()
raw.reset_index(inplace=True)

#raw.set_index('date',inplace=True)
ts = raw['return'].copy()
forecast_ahead = 7
fixed_window = False

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
# Define a function for model fitting and evaluation
def fit_evaluate_garch(p_order, q_order, o_order,mean_type, vol_type ,dist_type ):
    model = arch.arch_model(y = ts, vol= vol_type ,p = p_order, q = q_order,
                            o = o_order, mean = mean_type , dist = dist_type)
    results = model.fit(disp='off')
    return {"aic": results.aic,
            "bic": results.bic,
            "loglikelihood": results.loglikelihood}

## Backtesting with MAE, MSE
def evaluate(observation, forecast):
    # Call sklearn function to calculate MAE
    mae = mean_absolute_error(observation, forecast)
    print('Mean Absolute Error (MAE): {:.3g}'.format(mae))
    # Call sklearn function to calculate MSE
    mse = mean_squared_error(observation, forecast)
    print('Mean Squared Error (MSE): {:.3g}'.format(mse))
    return mae, mse

# Define the parameter grid
p_values = range(1,4)
q_values = range(3)
o_values = range(1)
mean_type_values = ['constant', 'zero', 'AR']
distribution_values = ['normal', 'skewt', 't']
vol_values = ['GARCH', 'EGARCH']
parameter_grid = list(product(p_values, q_values, o_values,
                              mean_type_values,
                              vol_values,
                              distribution_values))

parameter_grid = pd.DataFrame(parameter_grid,columns= ['p_order','q_order','o_order',
                                                       'mean_type_values',
                                                       'vol_type_values',
                                                       'distribution_values'])
# Initialize variables to store the best results
best_params = None
results_df = pd.DataFrame()

# Perform grid search
for i in range(parameter_grid.shape[0]):
    try:
        p_order, q_order, o_order, mean_type, vol_type, dist_type = parameter_grid.iloc[i, :]
        p_order, q_order, o_order = int(p_order),int(q_order),int(o_order)
        candidate = fit_evaluate_garch(p_order, q_order, o_order ,mean_type, vol_type ,dist_type)
        results_df = results_df.append({'aic': candidate['aic'],
                                        'sbc': candidate['bic'],
                                        'loglikelihood': candidate['loglikelihood']},
                          ignore_index=True)
        print(f'Itteration {i} of {parameter_grid.shape[0]} ')
    except:
        continue

# multiple ranking on performance criteria
results_df['rank_aic'] = results_df['aic'].rank()
results_df['rank_sbc'] = results_df['sbc'].rank()
results_df['rank_loglik'] = results_df['loglikelihood'].rank(ascending=False)
results_df['rank_total'] = results_df.eval('(rank_aic + rank_sbc + rank_loglik) /3 ')

# best possible model
best_pos = results_df['rank_total'].argmin()
results_df.drop(['rank_aic', 'rank_sbc', 'rank_loglik','rank_total'], axis=1, inplace=True)
best_p, best_q, best_o, best_mean, best_vol, best_dist = parameter_grid.iloc[best_pos,:] # here check for skiped models

# Fit the best model to the full data
best_p, best_q, best_o = int(best_p), int(best_q), int(best_o)
best_model = arch.arch_model(ts, vol=best_vol, p=best_p, q=best_q, o = best_o,
                             mean=best_mean, dist=best_dist)
best_results = best_model.fit(disp='off')

# Visualize results
# best_results.plot().show()


if fixed_window:
    ts.index = raw.date
    index = ts.index
    start_loc = 0
    end_loc = len(index) - forecast_ahead
    forecasts = {}
    for i in range(len(ts) - end_loc):
        sys.stdout.write('o')
        sys.stdout.flush()
        res = best_model.fit(first_obs=start_loc + i, last_obs=i + end_loc, disp='off')
        temp = res.forecast(horizon=1, start=end_loc).variance
        fcast = temp.iloc[i]
        forecasts[fcast.name] = fcast
    print(' Done!')
    variance_holdout = pd.DataFrame(forecasts).T
    variance_holdout = pd.Series(variance_holdout['h.1'])
else:
    ts.index = raw.date
    index = ts.index
    start_loc = 0
    end_loc = len(index) - forecast_ahead
    forecasts = {}
    for i in range(len(ts) - end_loc):
        sys.stdout.write('-')
        sys.stdout.flush()
        res = best_model.fit(first_obs=start_loc, last_obs=i + end_loc, disp='off')
        temp = res.forecast(horizon=1, start=end_loc).variance
        fcast = temp.iloc[i]
        forecasts[fcast.name] = fcast
    print(' Done!')
    variance_holdout = pd.DataFrame(forecasts).T
    variance_holdout = pd.Series(variance_holdout['h.1'])
    #print(best_results.summary())


# Backtest model with MAE, MSE
observed_var = raw['return'].tail(forecast_ahead).sub(raw['return'].tail(forecast_ahead).mean()).pow(2)
evaluate(observed_var, variance_holdout)

'''
# Plot the actual  volatility
bm_std = best_results.conditional_volatility
plt.plot(raw['return'].sub(raw['return'].mean()).pow(2),
         color = 'grey', alpha = 0.4, label = 'Daily Volatility')

# Plot EGARCH  estimated volatility
plt.plot(bm_std**2, color = 'red', label = 'Best Model Volatility')
plt.legend(loc = 'upper right')
plt.show()
'''

if best_vol == 'EGARCH':
    forecast = best_results.forecast(horizon=forecast_ahead, method="bootstrap").variance.T
else:
    forecast = best_results.forecast(horizon=forecast_ahead).variance.T

conf = np.sqrt(forecast)*1.96
