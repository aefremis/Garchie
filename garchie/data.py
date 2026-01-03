class crypto:
    """
    A class used to represent an asset in decentralized networks - cryptocurrencies, fetching data from Alpha Vantage.

    ...
    Attributes
    ----------
    symbol : str
        A string that corresponds to the asset's symbol (e.g., 'BTC').
    market : str
        The market/currency to compare against (e.g., 'USD').
    granularity : str
        A string that points to the time aggregation (e.g., "1d", "1wk", "1mo").
    start : str
        A string that denotes the start date in the format (yyyy-mm-dd).
    end : str
        A string that denotes the end date in the format (yyyy-mm-dd).
    """

    def __init__(self, symbol, market, granularity, start, end):
        """
        Parameters
       ----------
        symbol : str
            A string that corresponds to the asset's symbol.
        market : str
            The physical currency of the digital currency price, e.g., 'USD'.
        granularity : str
            A string that points to the time aggregation. Supported intervals are ["1d", "1wk", "1mo"].
        start : str
            A string that denotes the start date in the format (yyyy-mm-dd).
        end : str
            A string that denotes the end date in the format (yyyy-mm-dd).
        """
        self.symbol = symbol
        self.market = market
        self.granularity = granularity
        self.start = start
        self.end = end


    def __str__(self):
        return f"class of crypto object '{self.symbol}' "


    def fetch_crypto(self):
        """Gets the performance metrics of a cryptocurrency for a specified time window and aggregation level using Alpha Vantage.

        A valid Alpha Vantage API key is required.

        Parameters
        ----------
        self : an object of class crypto
            An object of class 'crypto' with relevant attributes.

        Returns
        -------
        DataFrame
            A DataFrame including performance metrics of the selected cryptocurrency.
            Returns an empty DataFrame if data fetching fails or no data is found.
        """
        import pandas as pd
        from alpha_vantage.cryptocurrencies import CryptoCurrencies

        # --- ACTION REQUIRED ---
        # Replace "YOUR_API_KEY" with the key you obtained from Alpha Vantage
        api_key = "YOUR_API_KEY"

        cc = CryptoCurrencies(key=api_key, output_format='pandas')

        try:
            if self.granularity == '1d':
                raw, meta_data = cc.get_digital_currency_daily(symbol=self.symbol, market=self.market)
            elif self.granularity == '1wk':
                raw, meta_data = cc.get_digital_currency_weekly(symbol=self.symbol, market=self.market)
            elif self.granularity == '1mo':
                raw, meta_data = cc.get_digital_currency_monthly(symbol=self.symbol, market=self.market)
            else:
                raise ValueError("Invalid granularity for crypto. Use '1d', '1wk', or '1mo'.")

        except Exception as e:
            print(f"Error fetching crypto data from Alpha Vantage: {e}")
            print("Please ensure your API key is correct and that the symbol and market are valid.")
            return pd.DataFrame()

        # Filter data based on start and end dates
        raw.index = pd.to_datetime(raw.index)
        raw = raw[(raw.index >= self.start) & (raw.index <= self.end)]

        if raw.empty:
            print(f"Warning: No crypto data found for {self.symbol} in the specified date range.")
            return raw

        # Standardize column names
        raw.columns = [col.split(' ')[-1] for col in raw.columns]
        
        # pct_change() expects specific column names, so we need to rename them
        raw.rename(columns={
            f'open({self.market.lower()})': 'open',
            f'high({self.market.lower()})': 'high',
            f'low({self.market.lower()})': 'low',
            f'close({self.market.lower()})': 'close',
            'volume': 'volume'
        }, inplace=True)
        
        raw.sort_index(inplace=True)
        raw['return'] = 100 * (raw['close'].pct_change())
        raw.dropna(axis=0, inplace=True)

        raw.reset_index(inplace=True)
        raw.rename(columns={'index':'date'}, inplace=True)

        return (df)


'''
sample use of class crypto

p1 = crypto(symbol='btc',granularity='1d', start= '2023-01-01', end='2023-06-25')
print(p1)
p1.fetch_crypto()
'''

class asset:
    """
        A class used to represent an asset in the stock markets, fetching data from Alpha Vantage.

        ...
        Attributes
        ----------
        symbol : str
            A string that corresponds to the asset's symbol (e.g., 'IBM', 'VUSA.AS').
        granularity : str
            A string that points to the time aggregation (e.g., "1d", "1wk", "1mo").
        start : str
            A string that denotes the start date in the format (yyyy-mm-dd) for filtering the data.
        end : str
            A string that denotes the end date in the format (yyyy-mm-dd) for filtering the data.

        Methods
        -------
        fetch_asset()
            Creates a DataFrame with performance metrics using Alpha Vantage.
        """
    def __init__(self, symbol, granularity, start, end):
        """
        Parameters
       ----------
        symbol : str
            A string that corresponds to the asset's symbol.
        granularity : str
            A string that points to the time aggregation. Supported intervals are ["1d", "1wk", "1mo"] for daily, weekly, and monthly respectively.
        start : str
            A string that denotes the start date in the format (yyyy-mm-dd).
        end : str
            A string that denotes the end date in the format (yyyy-mm-dd).
        """
        self.symbol = symbol
        self.granularity = granularity
        self.start = start
        self.end = end

    def __str__(self):
        return f"class of stock object '{self.symbol}' "

    def fetch_asset(self):
        """Gets the performance metrics of a symbol for a specified time window and aggregation level using Alpha Vantage.

        This method uses the Alpha Vantage API to fetch daily, weekly, or monthly data.
        A valid Alpha Vantage API key is required.
        - Daily ('1d') data is limited to the last 100 data points on the free tier.
        - Weekly ('1wk') and Monthly ('1mo') data fetches the full available history.

        Parameters
        ----------
        self : an object of class asset
            An object of class 'asset' with relevant attributes.

        Returns
        -------
        DataFrame
            A DataFrame including performance metrics of the selected symbol.
            Returns an empty DataFrame if data fetching fails or no data is found.
        """
        import pandas as pd
        from alpha_vantage.timeseries import TimeSeries
        
        # --- ACTION REQUIRED ---
        # Replace "YOUR_API_KEY" with the key you obtained from Alpha Vantage
        api_key = "OVAYAKMDMRYMDVUU"
        
        ts = TimeSeries(key=api_key, output_format='pandas')
        
        try:
            match self.granularity:
                case '1d':
                    raw, meta_data = ts.get_daily(symbol=self.symbol, outputsize='compact')
                case '1wk':
                    raw, meta_data = ts.get_weekly(symbol=self.symbol)
                case '1mo':
                    raw, meta_data = ts.get_monthly(symbol=self.symbol)
                case _:
                    raise ValueError("Invalid granularity specified. Use '1d', '1wk', or '1mo'.")

        except Exception as e:
            print(f"Error fetching data from Alpha Vantage: {e}")
            print("Please ensure your API key is correct and that the symbol and granularity are valid.")
            return pd.DataFrame() # Return empty dataframe on error

        # Filter data based on start and end dates
        raw.index = pd.to_datetime(raw.index)
        raw = raw[(raw.index >= self.start) & (raw.index <= self.end)]

        if raw.empty:
            print(f"Warning: No data found for {self.symbol} in the specified date range.")
            return raw

        raw.rename(columns={
            '1. open': 'open', 
            '2. high': 'high', 
            '3. low': 'low', 
            '4. close': 'close', 
            '5. volume': 'volume'
        }, inplace=True)
        raw.columns = raw.columns.str.lower()
        
        # The API might return slightly different column names for weekly/monthly (e.g. '5. volume' vs 'volume')
        # This standardizes them by taking the last word.
        raw.columns = [col.split(' ')[-1] for col in raw.columns]
        
        raw.sort_index(inplace=True)
        raw['return'] = (100 * raw['close'].pct_change())
        raw.dropna(axis = 0, inplace= True)
        
        # Reset index to have 'date' as a column, matching previous format
        raw.reset_index(inplace=True)
        raw.rename(columns={'index':'date'}, inplace=True)

        return(raw)

'''
sample use of class asset
p2 = asset(symbol='VUSA.AS',granularity='1d', start= '2024-01-01', end='2024-06-25')
print(p2)
p2.fetch_asset()
'''

class commodity:
    """
    A class to fetch commodity data from Alpha Vantage.

    ...
    Attributes
    ----------
    symbol : str
        The commodity to fetch. Supported symbols include 'WTI', 'BRENT', 'NATURAL_GAS', 
        'COPPER', 'ALUMINUM', 'WHEAT', 'CORN', 'COTTON', 'SUGAR', 'COFFEE', 'ALL_COMMODITIES'.
    granularity : str
        The time interval for the data. Supported intervals vary by commodity but
        generally include 'weekly', 'monthly', and 'quarterly'.
    start : str
        A string that denotes the start date in the format (yyyy-mm-dd) for filtering.
    end : str
        A string that denotes the end date in the format (yyyy-mm-dd) for filtering.
    """
    def __init__(self, symbol ,granularity, start, end):
        """
        Parameters
       ----------
        symbol : str
            The commodity symbol to fetch.
        granularity : str
            The time interval for the data (e.g., 'weekly', 'monthly').
        start : str
            The start date for filtering data, format 'yyyy-mm-dd'.
        end : str
            The end date for filtering data, format 'yyyy-mm-dd'.
        """
        self.symbol = symbol
        self.granularity = granularity
        self.start = start
        self.end = end


    def __str__(self):
        return f"class of commodity object '{self.symbol}' "


    def fetch_commodity(self):
        """
        Fetches commodity data from Alpha Vantage for a specified time window.

        This method maps the commodity symbol to the corresponding Alpha Vantage API call.
        A valid Alpha Vantage API key is required.

        Returns
        -------
        DataFrame
            A DataFrame including the price data for the selected commodity.
            Returns an empty DataFrame if data fetching fails.
        """
        import pandas as pd
        from alpha_vantage.commodities import Commodities

        # --- ACTION REQUIRED ---
        # Replace "YOUR_API_KEY" with the key you obtained from Alpha Vantage
        api_key = "YOUR_API_KEY"

        com = Commodities(key=api_key, output_format='pandas')
        
        try:
            commodity_map = {
                'WTI': com.get_wti,
                'BRENT': com.get_brent,
                'NATURAL_GAS': com.get_natural_gas,
                'COPPER': com.get_copper,
                'ALUMINUM': com.get_aluminum,
                'WHEAT': com.get_wheat,
                'CORN': com.get_corn,
                'COTTON': com.get_cotton,
                'SUGAR': com.get_sugar,
                'COFFEE': com.get_coffee,
                'ALL_COMMODITIES': com.get_all_commodities,
            }
            
            fetch_func = commodity_map.get(self.symbol.upper())
            
            if fetch_func:
                raw, meta_data = fetch_func(interval=self.granularity)
            else:
                raise ValueError(f"Invalid commodity symbol specified: {self.symbol}")

        except Exception as e:
            print(f"Error fetching commodity data from Alpha Vantage: {e}")
            print("Please ensure your API key is correct and that the symbol and interval are valid.")
            return pd.DataFrame()

        print("--- (DEBUG) Raw data from Alpha Vantage ---")
        print(raw.head())
        print(raw.tail())
        print("-----------------------------------------")

        # Filter data based on start and end dates
        raw.index = pd.to_datetime(raw.index)
        raw = raw[(raw.index >= self.start) & (raw.index <= self.end)]
        
        if raw.empty:
            print(f"Warning: No commodity data found for {self.symbol} in the specified date range.")
            return raw

        raw.rename(columns={'value': 'close'}, inplace=True)
        raw['close'] = pd.to_numeric(raw['close'], errors='coerce') # Ensure 'close' is numeric
        raw.sort_index(inplace=True)
        raw['return'] = (100 * raw['close'].pct_change())
        raw.dropna(axis=0, inplace=True)

        raw.reset_index(inplace=True)
        raw.rename(columns={'index':'date'}, inplace=True)
        
        return(raw)


'''
#sample use of class commodity
p3 = commodity(symbol= 'WTI', granularity = 'monthly', start =  '2022-01-01',  end = '2023-06-25')
print(p3)
raw = p3.fetch_commodity()
'''

