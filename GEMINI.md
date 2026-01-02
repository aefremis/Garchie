# Gemini Project Context: Financial Analysis Toolkit

## Project Overview

This project is a Python-based toolkit for financial time series analysis and forecasting. It provides a collection of scripts to fetch, analyze, and model financial data for various asset classes, including stocks, cryptocurrencies, and commodities. The primary goal is to build and evaluate different forecasting models to predict future asset prices and volatility.

The project utilizes several popular data science and machine learning libraries, such as pandas, statsmodels, scikit-learn, prophet, and lightgbm.

## File Descriptions

*   `fetch_equity.py`: Contains Python classes (`asset`, `crypto`, `commodity`) for fetching financial data from various APIs, including Yahoo Finance, Messari, and the Commodities API.
*   `descript_equity.py`: A script for performing descriptive and exploratory data analysis on financial time series. It generates various plots like moving averages, Bollinger Bands, and seasonality decompositions to help understand the data.
*   `garch_model.py`: Implements a GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model to analyze and forecast the volatility of asset returns. It includes a grid search functionality to find the optimal model parameters.
*   `gam_model.py`: Implements a GAM (Generalized Additive Model) using the Prophet library for time series forecasting.
*   `light_gbm.py`: Implements a LightGBM model for mean prediction, including feature engineering and hyperparameter tuning.
*   `mean_model.py`: Implements an autoregressive (AR) model for mean prediction, using `pmdarima` to find the optimal model order.
*   `theory/`: This directory contains PDF documents related to the theoretical concepts behind the models used in this project.

## Building and Running

This project is a collection of Python scripts and does not have a centralized build process. To run the analysis, you should execute the individual Python scripts.

**Prerequisites:**

The project relies on several Python libraries. While there is no `requirements.txt` file, you can infer the dependencies from the `import` statements in each script. You will need to install libraries such as:

*   `pandas`
*   `numpy`
*   `statsmodels`
*   `scikit-learn`
*   `prophet`
*   `lightgbm`
*   `pmdarima`
*   `yfinance`
*   `requests`
*   `matplotlib`
*   `seaborn`
*   `plotly`
*   `hampel`

It is recommended to use a Python virtual environment to manage dependencies. This project contains configurations for `.venv` and `myenv`.

**Running a Script:**

To run a specific analysis, you can execute the corresponding Python file. For example, to run the descriptive analysis script:

```bash
python descript_equity.py
```

You can modify the parameters within each script (e.g., the asset symbol, date range) to perform analysis on different data.

## Development Conventions

*   **Code Structure:** The project is organized into modules with specific responsibilities. Data fetching logic is encapsulated in `fetch_equity.py`. Each modeling approach (GARCH, GAM, etc.) has its own dedicated file.
*   **Modeling:** The modeling scripts are generally structured as classes that take time series data as input and provide methods for designing and evaluating the models.
*   **Visualization:** The project uses `matplotlib`, `seaborn`, and `plotly` for generating a variety of plots to visualize data and model results.
*   **Experimentation:** The scripts are set up for experimentation, with sample usage code provided at the end of each file (often commented out). This allows for easy modification and execution of different analyses.
