# Theta Model Documentation

## Overview
The `theta_model` class implements the Theta method, a simple, computationally efficient, and highly effective time series forecasting technique. It relies on the `ThetaModel` implementation from `statsmodels`.

## Usage
The model expects a pandas DataFrame with `date` and `typical_price` columns.

```python
from garchie.models.theta_model import theta_model

# Initialize the model
tm = theta_model(
    ts=ts_data, 
    train_size=0.8, 
    forecast_ahead=12, 
    diagnostics=True
)

# Run the forecasting pipeline
predictions = tm.design_theta_model()
```

## Key Components
- **Seasonality Detection:** Includes a robust internal method (`_get_seasonal_period`) that detrends the data and uses the Autocorrelation Function (ACF) to identify the highest correlating lag. It includes logic to snap raw lag numbers to common business/calendar cycles (e.g., 5 days for a business week, 12 for months).
- **Theta Decomposition:** Fits the `ThetaModel` using the discovered seasonal period.
- **Validation & Retraining:** Operates by doing a train/test split to generate validation intervals, followed by a complete retrain on the full dataset to generate the definitive future predictions.

## Returns
A standardized DataFrame containing future price predictions:
`['date', 'prediction', 'model_name', 'variable', 'lower_bound', 'upper_bound']`