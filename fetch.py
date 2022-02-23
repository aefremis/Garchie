import numpy as np
import pandas as pd
import requests
from arch import arch_model

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.interactive(False)


def get_crypto_price(symbol, start, end, granularity):
    api_url = f'https://data.messari.io/api/v1/markets/binance-{symbol}-usdt/metrics/price/time-series?start={start}&end={end}&interval={granularity}'
    raw = requests.get(api_url).json()
    df = pd.DataFrame(raw['data']['values'])
    df = df.rename(columns={0: 'date', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df.set_index('date')
    return df


def simulate_GARCH(n, omega, alpha, beta=0):
    np.random.seed(4)
    # Initialize the parameters
    white_noise = np.random.normal(size=n)
    resid = np.zeros_like(white_noise)
    variance = np.zeros_like(white_noise)

    for t in range(1, n):
        # Simulate the variance (sigma squared)
        variance[t] = omega + alpha * resid[t - 1] ** 2 + beta * variance[t - 1]
        # Simulate the residuals
        resid[t] = np.sqrt(variance[t]) * white_noise[t]

    return resid, variance

crypto = get_crypto_price('xrp', '2022-01-01', '2022-02-15', '1d')
crypto = crypto.sort_index()

# Calculate daily returns as percentage price changes
crypto['Return'] = 100 * (crypto['close'].pct_change())

# drop non volatility observations
crypto.dropna(axis='rows',
              inplace=True)

# keep only close value of the day in a dataframe
crypto = crypto[['Return']]


# plot the data
plt.plot(crypto['Return'], color = 'tomato', label = 'Daily Returns')
plt.legend(loc='upper right')
plt.show()

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

# simulated models and forecasts
# Simulate a ARCH(1) series
arch_resid, arch_variance = simulate_GARCH(n= 200,omega = 0.1, alpha = 0.7)
# Simulate a GARCH(1,1) series
garch_resid, garch_variance = simulate_GARCH(n= 200,
                                             omega = 0.1, alpha = 0.7,
                                             beta = 0.1)

# Plot the ARCH variance
plt.plot(arch_variance, color = 'red', label = 'ARCH Variance')
# Plot the GARCH variance
plt.plot(garch_variance, color = 'orange', label = 'GARCH Variance')
plt.legend()
plt.show()

