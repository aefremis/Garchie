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

p1 = crypto(symbol='btc',granularity='1w', start= '2022-01-01', end='2022-11-29')
print(p1)
p1.fetch_crypto()
'''

class asset:
    """
        A class used to represent an asset in the stock markets

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
        fetch_asset()
            Creates a DataFrame with performance metrics
        """
    def __init__(self, symbol, granularity, start, end):
        """
        Parameters
       ----------
        symbol : str
        a string that corresponds to the asset's symbol
        granularity : str
        a sting that points to the time aggregation. Supported intervals are ["d", "m", "w"] for daily, monthly, weekly respectively.
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
        """Gets the performance metrics of a symbol for a time window and aggregation

        Parameters
        ----------
        self : an object of class asset
            An object of class 'asset' with relevant attributes

        Returns
        -------
        DataFrame
            a DataFrame including performance metrics of selected symbol
        """
        import pandas as pd
        from pandas_datareader import data as stc
        raw = stc.get_data_yahoo(symbols=self.symbol,
                                 start=self.start,
                                 end=self.end,
                                 interval=self.granularity)
        raw.drop('Adj Close', axis=1, inplace=True)
        raw.reset_index(inplace=True)
        raw.columns = raw.columns.str.lower()
        raw['date'] = pd.to_datetime(raw['date'], unit='ms')
        raw.set_index('date').sort_index()
        raw['return'] = (100 * raw['close'].pct_change())
        raw.dropna(axis = 0, inplace= True)
        return(raw)

'''
sample use of class asset
p2 = asset(symbol='CT=F',granularity='d', start= '2022-01-01', end='2022-11-29')
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

        access_key = '4rsap4p3c2o365t01lyf8eho0wjpwdgz7z1d8t1rt48txpowp8giivv0z278'
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
p3 = commodity(base_currency='USD', symbol= 'CORN',granularity = 'w',start =  '2022-05-01',  end = '2022-11-30')
print(p3)
p3.fetch_commodity()
'''
