import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import hampel as hm

class EDA:
    """
    A generic class for Exploratory Data Analysis (EDA) and plotting of financial time series.
    """
    def __init__(self, df):
        """
        Initialize the EDA class.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing financial data. Must have a 'close' column.
            If 'date' column is present, it will be set as index.
        """
        self.df = df.copy()
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df.set_index('date', inplace=True)
        
        # Standardize column names to lower case for internal use if needed, 
        # but for now assuming standard 'close', 'open', 'high', 'low'.
        # Ensure 'close' exists
        if 'close' not in self.df.columns:
            # Try to handle case-insensitive check or different naming convention if needed
            # For now strict check
            raise ValueError("Dataframe must contain 'close' column")

    @staticmethod
    def plot_gam_diagnostics(model, forecast):
        """
        Plot GAM (Prophet) diagnostics using Plotly and Matplotlib.
        
        Parameters
        ----------
        model : prophet.Prophet
            Fitted Prophet model.
        forecast : pd.DataFrame
            Prophet forecast dataframe.
        """
        try:
            from prophet.plot import plot_plotly, plot_components_plotly

            plot_plotly(model, forecast).show()
            plot_components_plotly(model, forecast).show()
            
        except ImportError:
            print("Prophet is not installed or import failed inside EDA.")
        except Exception as e:
            print(f"Error plotting GAM diagnostics: {e}")

    @staticmethod
    def plot_mean_model_diagnostics(train, test, validation_df=None, forecast_df=None, model=None):
        """
        Plot Mean Model (ARIMA/SARIMA) diagnostics.

        Parameters
        ----------
        train : pd.Series
            Training data series.
        test : pd.Series
            Testing data series.
        validation_df : pd.DataFrame, optional
            DataFrame containing validation predictions with columns ['pred', 'lower', 'upper'].
            Index should match the test series.
        forecast_df : pd.DataFrame, optional
            DataFrame containing future forecast with columns ['prediction', 'lower_bound', 'upper_bound', 'date'].
        model : pmdarima.arima.arima.ARIMA, optional
            Fitted ARIMA model object for residual plots.
        """
        # 1. Train/Test Split Visualization
        if validation_df is None and forecast_df is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(train, label='Train', marker='.')
            ax.plot(test, label='Test', marker='.')
            ax.set_title('Train/Test Split')
            ax.legend()
            plt.show()
            return

        # 2. Model Diagnostics (Residuals)
        if model is not None:
            try:
                model.plot_diagnostics(figsize=(12, 8))
                plt.suptitle('Model Diagnostics (Residuals)')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not plot model diagnostics: {e}")

        # 3. Validation Plot (if provided)
        if validation_df is not None:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(train, label='Train', marker='.')
            ax.plot(test, label='Test', marker='.')
            
            if 'pred' in validation_df.columns:
                ax.plot(validation_df.index, validation_df['pred'], label='Validation Prediction', ls='--', linewidth=2, color='green')
            
            if 'lower' in validation_df.columns and 'upper' in validation_df.columns:
                ax.fill_between(validation_df.index, validation_df['lower'], validation_df['upper'], color='green', alpha=0.15, label='95% CI')

            ax.set_title('Model Validation (Backtest)', fontsize=16)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 4. Future Forecast Plot (if provided)
        if forecast_df is not None:
            fig, ax = plt.subplots(figsize=(12, 7))
            # Plot recent history for context
            combined_history = pd.concat([train, test])
            
            # Ensure index is datetime for proper plotting alongside forecast dates
            if not isinstance(combined_history.index, pd.DatetimeIndex):
                 # Try to convert if it looks like dates
                 try:
                     combined_history.index = pd.to_datetime(combined_history.index)
                 except:
                     pass

            ax.plot(combined_history.index, combined_history.values, label='History', marker='.', color='black', alpha=0.5)
            
            if 'prediction' in forecast_df.columns:
                # Ensure date is available for x-axis
                if 'date' in forecast_df.columns:
                    x_axis = pd.to_datetime(forecast_df['date'])
                else:
                    x_axis = pd.to_datetime(forecast_df.index)
                
                ax.plot(x_axis, forecast_df['prediction'], label='Future Forecast', ls='--', linewidth=3, color='red')

                if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                    ax.fill_between(x_axis, forecast_df['lower_bound'], forecast_df['upper_bound'], color='red', alpha=0.15, label='95% Forecast CI')

            ax.set_title('Future Forecast', fontsize=16)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_garch_diagnostics(ts, best_results=None, title="Historical Volatility"):
        """
        Plot GARCH diagnostics: historical volatility proxy (returns squared) vs estimated volatility.

        Parameters
        ----------
        ts : pd.Series
            Time series of returns.
        best_results : arch.univariate.base.ARCHModelResult, optional
            The fitted model results object from the 'arch' library.
        title : str
            Title of the plot.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot proxy volatility (squared returns)
        # Using sub(mean) to center, though often returns are assumed mean zero in simplistic volatility checks
        vol_proxy = ts.sub(ts.mean()).pow(2)
        plt.plot(vol_proxy.index, vol_proxy, color='grey', alpha=0.4, label='Daily Volatility (Returns^2)')

        if best_results is not None:
            # Conditional volatility from the model (sigma)
            # Typically we plot sigma^2 (variance) to match returns^2
            estimated_variance = best_results.conditional_volatility**2
            plt.plot(estimated_variance.index, estimated_variance, color='red', label='Estimated Model Variance', linewidth=1.5)

        plt.title(title, fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Variance')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_smoothed(self, window=14, title="Smoothed Price", save_path=None):
        """
        Plot the close price and a simple moving average.
        """
        plt.figure(figsize=(12, 6))
        sma = self.df['close'].rolling(window=window).mean()
        plt.plot(self.df.index, self.df['close'], label='Close Price', alpha=0.5)
        plt.plot(self.df.index, sma, label=f'SMA ({window})', linewidth=2)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_bollinger_bands(self, window=20, num_std=2, save_path=None):
        """
        Calculate and plot Bollinger Bands.
        """
        sma = self.df['close'].rolling(window=window).mean()
        std = self.df['close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['close'], label='Close Price', color='navy', alpha=0.6)
        plt.plot(self.df.index, sma, label=f'SMA ({window})', color='teal', alpha=0.8)
        plt.plot(self.df.index, upper_band, label='Upper Band', color='red', alpha=0.5, linestyle='--')
        plt.plot(self.df.index, lower_band, label='Lower Band', color='green', alpha=0.5, linestyle='--')
        plt.fill_between(self.df.index, lower_band, upper_band, color='gray', alpha=0.1)
        
        plt.title(f'Bollinger Bands (Window={window}, Std={num_std})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_rsi(self, window=14, save_path=None):
        """
        Calculate and plot Relative Strength Index (RSI).
        """
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, rsi, label=f'RSI ({window})', color='purple')
        plt.axhline(70, linestyle='--', alpha=0.5, color='red', label='Overbought (70)')
        plt.axhline(30, linestyle='--', alpha=0.5, color='green', label='Oversold (30)')
        plt.title(f'Relative Strength Index (RSI)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_seasonal_decomposition(self, period=30, model='multiplicative', save_path=None):
        """
        Perform and plot seasonal decomposition using statsmodels.
        """
        # Ensure no NaNs and strictly positive for multiplicative
        df_clean = self.df['close'].dropna()
        if model == 'multiplicative' and (df_clean <= 0).any():
             print("Data has non-positive values, switching to additive model for decomposition.")
             model = 'additive'

        decomposition = seasonal_decompose(df_clean, model=model, period=period)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        decomposition.observed.plot(ax=axes[0], title='Observed', color='navy')
        axes[0].set_ylabel('Observed')
        
        decomposition.trend.plot(ax=axes[1], title='Trend', color='teal')
        axes[1].set_ylabel('Trend')
        
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='green')
        axes[2].set_ylabel('Seasonal')
        
        decomposition.resid.plot(ax=axes[3], title='Residual', color='red', style='.')
        axes[3].set_ylabel('Residual')
        
        plt.xlabel('Date')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_candlestick(self, save_path=None):
        """
        Plot an interactive candlestick chart using Plotly.
        """
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.df.columns for col in required_cols):
            print(f"Missing required columns for candlestick: {required_cols}")
            return

        fig = go.Figure(data=[go.Candlestick(
            x=self.df.index,
            open=self.df['open'],
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close']
        )])
        
        fig.update_layout(
            title='Candlestick Chart',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white'
        )
        
        if save_path:
            # If path ends in .html, save as html, else try static image (requires kaleido)
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                try:
                    fig.write_image(save_path)
                except ValueError as e:
                    print(f"Could not save static image (install kaleido?): {e}. Saving as HTML instead.")
                    fig.write_html(save_path + ".html")
        else:
            fig.show()

    def plot_volatility(self, window=30, save_path=None):
        """
        Calculate and plot rolling volatility (standard deviation of returns).
        """
        # Calculate returns
        returns = self.df['close'].pct_change().dropna()
        
        # Rolling standard deviation (volatility)
        rolling_vol = returns.rolling(window=window).std()
        
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_vol.index, rolling_vol, color='orange', label=f'{window}-Period Rolling Volatility')
        plt.title(f'Rolling Volatility (Window={window})')
        plt.xlabel('Date')
        plt.ylabel('Volatility (Std Dev of Returns)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def apply_hampel_filter(self, window=7, n_sigmas=3, save_path=None):
        """
        Apply Hampel filter to detect and plot outliers.
        """
        try:
            # The hampel function returns a Result object with outlier_indices
            # n_sigma must be a float
            result = hm.hampel(self.df['close'], window_size=window, n_sigma=float(n_sigmas))
            outlier_indices = result.outlier_indices
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.df.index, self.df['close'], color='navy', label='Close Price', alpha=0.6)
            
            if len(outlier_indices) > 0:
                # Map integer indices back to datetime index
                outlier_dates = self.df.iloc[outlier_indices].index
                outlier_values = self.df.iloc[outlier_indices]['close']
                
                plt.scatter(outlier_dates, outlier_values, color='red', s=50, label='Outliers (Hampel)', zorder=5)
            
            plt.title(f'Hampel Filter Outlier Detection (Window={window}, Sigma={n_sigmas})')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
            
        except Exception as e:
            print(f"Error applying Hampel filter: {e}")

    def plot_forecast_results(self, validation_df, future_df, weights, save_path=None):
        """
        Plot historical data, validation performance, and future weighted forecast with confidence intervals.
        
        Parameters
        ----------
        validation_df : pd.DataFrame
            DataFrame containing validation results. Must contain 'date', 'gam_pred', 'mean_pred', 'weighted_forecast', 'lower_ci', 'upper_ci'.
        future_df : pd.DataFrame
            DataFrame containing future forecast results. Must contain 'date', 'gam_pred', 'mean_pred', 'weighted_forecast', 'lower_ci', 'upper_ci'.
        weights : dict
            Dictionary containing model weights (e.g., {'gam': 0.7, 'mean': 0.3}).
        save_path : str, optional
            Path to save the plot. If None, the plot is shown.
        """
        w_gam = weights.get('gam', 0)
        w_mean = weights.get('mean', 0)
        w_theta = weights.get('theta', 0)
        
        plt.figure(figsize=(14, 8))
        
        # 1. Plot Full History (Actuals) from self.df
        # Assuming 'typical_price' or 'close' is the target. Using 'close' as standard, 
        # but weighted_forecast uses 'typical_price'. We will check if typical_price exists, else close.
        col_to_plot = 'typical_price' if 'typical_price' in self.df.columns else 'close'
        
        plt.plot(self.df.index, self.df[col_to_plot], label='Actual History', color='gray', alpha=0.6, linewidth=1.5)
        
        # 2. Plot Validation Predictions (Backtest)
        # Ensure validation_df has 'date' column or index
        if 'date' in validation_df.columns:
            val_dates = pd.to_datetime(validation_df['date'])
        else:
            val_dates = validation_df.index

        # Plot individual model validation predictions if available
        if 'gam_pred' in validation_df.columns:
            plt.plot(val_dates, validation_df['gam_pred'], label='GAM Validation', color='blue', linestyle=':', linewidth=2, alpha=0.7)
        if 'mean_pred' in validation_df.columns:
            plt.plot(val_dates, validation_df['mean_pred'], label='Mean Model Validation', color='red', linestyle=':', linewidth=2, alpha=0.7)
        if 'theta_pred' in validation_df.columns:
            plt.plot(val_dates, validation_df['theta_pred'], label='Theta Model Validation', color='green', linestyle=':', linewidth=2, alpha=0.7)
        
        # Plot Weighted Validation Forecast
        if 'weighted_forecast' in validation_df.columns:
            plt.plot(val_dates, validation_df['weighted_forecast'], label='Weighted Validation Forecast', color='purple', linestyle='--', linewidth=2)
            
            # Plot GARCH Confidence Intervals for Validation
            if 'lower_ci' in validation_df.columns and 'upper_ci' in validation_df.columns:
                plt.fill_between(val_dates, validation_df['lower_ci'], validation_df['upper_ci'], 
                                 color='purple', alpha=0.15, label='95% GARCH Validation CI')
        
        # 3. Plot Future Forecasts
        if 'date' in future_df.columns:
            fut_dates = pd.to_datetime(future_df['date'])
        else:
            fut_dates = future_df.index

        if 'gam_pred' in future_df.columns:
            plt.plot(fut_dates, future_df['gam_pred'], label=f'GAM Forecast (Weight: {w_gam:.2f})', color='blue', linestyle='--', alpha=0.5)
        if 'mean_pred' in future_df.columns:
            plt.plot(fut_dates, future_df['mean_pred'], label=f'Mean Model Forecast (Weight: {w_mean:.2f})', color='red', linestyle='--', alpha=0.5)
        if 'theta_pred' in future_df.columns:
            plt.plot(fut_dates, future_df['theta_pred'], label=f'Theta Model Forecast (Weight: {w_theta:.2f})', color='green', linestyle='--', alpha=0.5)
        
        if 'weighted_forecast' in future_df.columns:
            plt.plot(fut_dates, future_df['weighted_forecast'], label='Combined Forecast', color='black', linewidth=2.5)
        
        # 4. Plot GARCH Confidence Intervals (Future)
        if 'lower_ci' in future_df.columns and 'upper_ci' in future_df.columns:
            plt.fill_between(fut_dates, future_df['lower_ci'], future_df['upper_ci'], 
                             color='orange', alpha=0.3, label='95% GARCH Forecast CI')
        
        # Add vertical lines for context
        last_hist_date = self.df.index[-1]
        plt.axvline(x=last_hist_date, color='green', linestyle='-', linewidth=1, label='Forecast Start')
        
        # Approximate validation start
        if len(val_dates) > 0:
            val_start_date = val_dates.iloc[0] if hasattr(val_dates, 'iloc') else val_dates[0]
            plt.axvline(x=val_start_date, color='orange', linestyle='-', linewidth=1, label='Validation Start')
        
        plt.xlim(left=self.df.index.min())

        plt.title(f'Forecast Results: History, Validation & Weighted Future with GARCH Bands', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            print(f"\nSaving plot to {save_path}...")
            plt.savefig(save_path)
            print("Plot saved successfully.")
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    # Test block
    # Create synthetic data
    dates = pd.date_range(start='2023-01-01', periods=100)
    # Ensure positive values for multiplicative decomposition
    data = pd.DataFrame({
        'date': dates,
        'open': np.abs(np.random.randn(100).cumsum() + 100),
        'close': np.abs(np.random.randn(100).cumsum() + 100),
        'high': np.abs(np.random.randn(100).cumsum() + 105),
        'low': np.abs(np.random.randn(100).cumsum() + 95)
    })
    
    eda = EDA(data)
    print("Testing EDA class...")
    # eda.plot_smoothed()
    # eda.plot_bollinger_bands()
