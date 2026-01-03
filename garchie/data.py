class crypto:
    """
    A class used to represent an asset in decentralized networks - cryptocurrencies

    ...
    Attributes
    ----------
    symbol : str
        a string that corresponds to the asset's symbol
    granularity : str
        a sting that points to the time aggregation
    start : str
        a string that denotes the start date in the format (yyyy-mm-dd)
    end : str
        a string that denotes the start date in the format (yyyy-mm-dd)

    Methods
    -------
    fetch_crypto()
        Creates a DataFrame with performance metrics
    """

    def __init__(self, symbol, granularity, start, end):
        """
        Parameters
       ----------
        symbol : str
        a string that corresponds to the asset's symbol
        granularity : str
        a sting that points to the time aggregation. Supported intervals are ["1h", "1d", "1w"] for 1 hour, 1 day, and 1 week respectively.
        start : str
        a string that denotes the start date in the format (yyyy-mm-dd)
        end : str
        a string that denotes the start date in the format (yyyy-mm-dd)
        """
        self.symbol = symbol
        self.granularity = granularity
        self.start = start
        self.end = end


    def __str__(self):
        return f"class of crypto object '{self.symbol}' "


    def fetch_crypto(self):
        """Gets the performance metrics of a symbol for a time window and aggregation

        Parameters
        ----------
        self : an object of class crypto
            An object of class 'crypto' with relevant attributes

        Returns
        -------
        DataFrame
            a DataFrame including performance metrics of selected symbol
        """
        import pandas as pd
        import requests
        api_url = f'https://data.messari.io/api/v1/markets/binance-{self.symbol}-usdt/metrics/price/time-series?start={self.start}&end={self.end}&interval={self.granularity}'
        raw = requests.get(api_url).json()
        df = pd.DataFrame(raw['data']['values'])
        df = df.rename(columns={0: 'date', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df = df.set_index('date').sort_index()
        df['return'] = 100 * (df['close'].pct_change())
        df.dropna(axis=0, inplace=True)
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
            A string that points to the time aggregation (e.g., "1d" for daily).
            Note: Alpha Vantage's free tier for daily data primarily uses 'compact' output,
            providing the last 100 data points, making the specified start/end dates
            potentially less effective for historical ranges beyond 100 days.
        start : str
            A string that denotes the start date in the format (yyyy-mm-dd).
        end : str
            A string that denotes the end date in the format (yyyy-mm-dd).

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
        a string that corresponds to the asset's symbol
        granularity : str
        a sting that points to the time aggregation. Supported intervals are ["1d", "1mo", "1wk"] for daily, monthly, weekly respectively.
        start : str
        a string that denotes the start date in the format (yyyy-mm-dd)
        end : str
        a string that denotes the start date in the format (yyyy-mm-dd)
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
        api_key = "YOUR_API_KEY"
        
        ts = TimeSeries(key=api_key, output_format='pandas')
        
        try:
            if self.granularity == '1d':
                raw, meta_data = ts.get_daily(symbol=self.symbol, outputsize='compact')
            elif self.granularity == '1wk':
                raw, meta_data = ts.get_weekly(symbol=self.symbol)
            elif self.granularity == '1mo':
                raw, meta_data = ts.get_monthly(symbol=self.symbol)
            else:
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
    A class used to represent a raw material or primary agricultural product
    ...
    Attributes
    ----------
    base_currency : str
        a string that corresponds to the commodity's base currency
    symbol : str
        a sting that corresponds to the commodity's symbol
    start : str
        a string that denotes the start date in the format (yyyy-mm-dd)
    end : str
        a string that denotes the start date in the format (yyyy-mm-dd)
    granularity : str
        a sting that points to the time aggregation. Supported intervals are ["d", "m", "w"] for daily, monthly, weekly respectively.

    Methods
    -------
    fetch_commodity()
        Creates a DataFrame with commodity's daily price
    """
    def __init__(self, base_currency, symbol ,granularity, start, end):

        """
        Parameters
       ----------
        base_currency : str
            a string that corresponds to the commodity's base currency
        symbol : str
            a sting that corresponds to the commodity's symbol
        start : str
            a string that denotes the start date in the format (yyyy-mm-dd)
        end : str
            a string that denotes the start date in the format (yyyy-mm-dd)
        granularity : str
            a sting that points to the time aggregation. Supported intervals are ["d", "m", "w"] for daily, monthly, weekly respectively.
        """
        self.base_currency = base_currency
        self.symbol = symbol
        self.granularity = granularity
        self.start = start
        self.end = end


    def __str__(self):
        return f"class of commodity object '{self.symbol}' "

    def time_roll(self, gran, data_roll):
        """
        Aggregates daily price of a commodity for a time level

        Parameters
        ----------
        gran : a sting that points to the time aggregation. Supported intervals are ["d", "m", "w"] for daily, monthly, weekly respectively.
        data_roll :  a DataFrame of daily commodity prices

        Returns
        -------
        DataFrame
        a DataFrame including aggregated price of a commodity for a time window
        """
        if gran == 'd':
            data_roll['return'] = (100 * data_roll.iloc[:, 0].pct_change())
            data_roll.dropna(axis=0, inplace=True)
        else:
            data_roll.reset_index(inplace=True)
            if gran == 'w':
                dyn_char, agg_level = '.dt.isocalendar().week', 'week'
            elif gran == 'm':
                dyn_char, agg_level = '.dt.month', 'month'

            exec("data_roll['"+str(agg_level)+"'] = data_roll['index']" + dyn_char)
            data_roll['year'] = data_roll['index'].dt.isocalendar().year
            data_roll =  eval("data_roll.groupby(['year', '"+str(agg_level)+"']).agg({'index': 'last', data_roll.columns[1]: 'last'})")
            data_roll.reset_index(drop=True, inplace=True)
            data_roll.set_index('index', inplace=True)
            data_roll['return'] = (100 * data_roll.iloc[:, 0].pct_change())
            data_roll.dropna(axis=0, inplace=True)
        return(data_roll)

    def fetch_commodity(self):
        """
        Gets the daily price of a commodity for a time window

        Parameters
        ----------
        self : an object of class commodity
            An object of class 'commodity' with relevant attributes

        Returns
        -------
        DataFrame
            a DataFrame including daily price of a commodity for a time window
        """

        import datetime as dt
        import requests
        import pandas as pd

        access_key = 'adg0rmqpjs8s5do6o5dzyx78i53u7k7q7mlsg1km1439eo0agw74yisikr8m'
        api_url = f'https://commodities-api.com/api/timeseries?access_key={access_key}&base={self.base_currency}&symbols={self.symbol}&start_date={self.start}&end_date={self.end}'

        raw = requests.get(api_url).json()
        df = pd.DataFrame(raw['data']['rates']).transpose()
        df.drop(df.columns[0],axis=1,inplace= True)
        df.index = pd.to_datetime(df.index, unit='ns')
        df.sort_index(inplace=True)
        df = self.time_roll(self.granularity, df.copy())
        return(df)



'''
#sample use of class commodity
p3 = commodity(base_currency='USD', symbol= 'WTI OIL',granularity = 'd',start =  '2023-01-01',  end = '2023-06-25')
print(p3)
raw = p3.fetch_commodity()
'''

