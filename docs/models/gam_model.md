# GAM Model Documentation

## Overview
The `gam_model` class implements a Generalized Additive Model (GAM) for time series forecasting, leveraging the `Prophet` library by Meta. It is designed to model the mean (typical price) of a financial asset by decomposing the time series into trend, seasonality, and holiday effects.

## Usage
The class is instantiated with a pandas DataFrame containing a `date` and `typical_price` column. 

```python
from garchie.models.gam_model import gam_model

# Initialize the model
gam = gam_model(
    ts=ts_data, 
    forecast_ahead=12, 
    forecast_unit='weeks', 
    diagnostics=True
)

# Run the design and forecasting pipeline
predictions = gam.design_gam_model()
```

## Key Components
- **Prophet Backend:** The model heavily relies on `prophet.Prophet`, which robustly handles missing data and large outliers.
- **Automated Design (`design_gam_model`):** This method prepares the DataFrame by renaming columns to Prophet's expected `ds` (datestamp) and `y` (target) formats, fits the model, and extends the dataframe into the future using `make_future_dataframe`.
- **Diagnostics:** When `diagnostics=True`, the model will leverage the `EDA` module to visually plot the training fit, test validation, and the future forecast.

## Returns
The pipeline returns a standardized DataFrame with the following columns:
`['date', 'prediction', 'model_name', 'variable', 'lower_bound', 'upper_bound']`