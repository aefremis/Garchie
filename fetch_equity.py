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
        a sting that points to the time aggregation
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
        df['Return'] = 100 * (df['close'].pct_change())
        df.dropna(axis=0, inplace=True)
        return (df)



p1 = crypto(symbol='sol',granularity='1d', start= '2022-10-01', end='2022-11-29')
print(p1)
p1.fetch_crypto()

