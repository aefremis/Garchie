# Weighted Ensemble Forecast

## Overview
The `weighted_forecast.py` script serves as a master pipeline that combines multiple predictive models (GAM, Mean/ARIMA, and Theta) into a single, robust ensemble forecast. It also layers a GARCH volatility model on top to generate dynamic, risk-adjusted confidence intervals.

## Ensemble Methodology

### 1. Validation & Backtesting
The data is split into a training set and a validation set (typically the last `N` periods).
Each constituent mean model (GAM, Mean, Theta) is trained strictly on the training set and tasked with predicting the validation period.

### 2. Inverse MAE Weighting
Once validation predictions are generated, they are compared against the actual historical values using Mean Absolute Error (MAE).
Weights for the ensemble are calculated inversely proportional to the error—meaning the model that performed best during the validation backtest is given the heaviest weighting in the final combined forecast.

```python
inv_gam = 1 / mae_gam
inv_mean = 1 / mae_mean
inv_theta = 1 / mae_theta
total_inv = inv_gam + inv_mean + inv_theta

w_gam = inv_gam / total_inv
# ... (applied to all models)
```

### 3. GARCH Confidence Intervals
Instead of relying on static standard deviations or the internal confidence intervals of a single model, a **GARCH(1,1)** model is run on the historical returns to forecast future volatility. This projected volatility is transformed into a standard deviation and mapped into a price-band envelope (scaled for visual plotting) around the ensemble's weighted mean prediction.

---

## Architectural Decision: Full Retraining vs. Parameter Reuse

A critical architectural decision in this pipeline is how the "Full Forecast" is generated after the validation phase.

**The Pipeline's Approach: Total Retraining**
When moving from the validation phase to generating the final future forecast, the pipeline completely re-instantiates and re-runs the `design_[model]()` methods for every algorithm using the entire dataset (Train + Validation data).

* **Why?** Time series data, especially in financial markets, is highly non-stationary and subject to regime shifts. The hyper-parameters that were optimal for a dataset ending 12 weeks ago (the validation split point) might no longer be the mathematical optimum when the most recent 12 weeks of market data are included. 
* By executing the entire grid search again (e.g., finding new ARIMA `p,d,q` terms or new GARCH `p,q` terms) on the full set, the pipeline guarantees that the model adapts to the absolute latest data dynamics.
* **Trade-off:** This approach favors maximum theoretical accuracy and adaptability over computational efficiency. It forces the system to perform heavy calculations (like Auto-ARIMA and GARCH grid searches) twice per run.

An alternative approach would be to lock the parameters discovered during validation and simply run a `predict()` forward, but the current `garchie` architecture deliberately re-optimizes to capture late-breaking trends.