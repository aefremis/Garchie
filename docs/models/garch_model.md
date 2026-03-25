# GARCH Model Documentation

## Overview
The `garch_model` class implements a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model to analyze and forecast the *volatility* (variance) of asset returns. It utilizes the `arch` Python library.

## Usage
Unlike the mean models that consume prices, the GARCH model is designed to consume **returns** (percentage changes).

```python
from garchie.models.garch_model import garch_model

# Initialize the model using a series of returns
gm = garch_model(
    ts=returns_series, 
    forecast_ahead=12, 
    fixed_window=False, 
    diagnostics=True
)

# Run the grid search and forecasting pipeline
volatility_forecast = gm.design_garch_model()
```

## Key Components
- **Grid Search Optimization:** The `design_garch_model` runs a comprehensive grid search across multiple hyperparameters:
  - `p` (lag order of symmetric innovation): 1 to 2
  - `q` (lag order of lagged volatility): 1 to 2
  - `o` (lag order of asymmetric innovation): 0 to 1
  - Mean models: 'Constant', 'Zero', 'AR'
  - Volatility models: 'GARCH', 'EGARCH'
  - Distributions: 'Normal', 't'
- **Model Selection:** The optimal model is chosen based on minimizing the Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC).
- **Backtesting & Forecasting:** The module provides functionality for rolling window backtesting and uses bootstrap or simulation methods to project future variance.
- **Output:** The returned prediction is the standard deviation (volatility), calculated as the square root of the forecasted variance.

## Returns
A standardized DataFrame containing the predicted future volatility:
`['date', 'prediction', 'model_name', 'variable', 'lower_bound', 'upper_bound']`