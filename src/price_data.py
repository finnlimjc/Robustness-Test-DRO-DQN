import numpy as np
import pandas as pd
import yfinance as yf

class YahooFinance:
    def __init__(self, symbol:str, start_date:str, end_date:str, interval:str='1d'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.col_names = ["date", "price", "normalized_price", 'log_price', 'log_return']
    
    def _download_data(self, symbol:str|list[str], start_date:str, end_date:str, interval:str='1d') -> pd.DataFrame:
        prices = yf.download(symbol, start=start_date, end=end_date, interval=interval, auto_adjust=False)
        prices.index = prices.index.date
        return prices
    
    def _filter_cols(self, prices:pd.DataFrame) -> pd.DataFrame:
        filtered_prices = prices['Adj Close'].copy()
        filtered_prices.columns = ['Adj Close']
        return filtered_prices
    
    def _normalize_price(self, filtered_prices:pd.DataFrame) -> pd.DataFrame:
        df = filtered_prices.copy()
        start_price = df.loc[0, 'Adj Close']
        df['normalised'] = df['Adj Close'] / start_price
        return df
    
    def _log_price(self, filtered_prices:pd.DataFrame) -> pd.DataFrame:
        df = filtered_prices.copy()
        df['log_price'] = np.log(df['Adj Close'])
        return df
    
    def _pct_change(self, filtered_prices:pd.DataFrame) -> pd.DataFrame:
        df = filtered_prices.copy()
        df['log_return'] = np.log(df['Adj Close']/ df['Adj Close'].shift(1))
        return df
    
    def _reset_index(self, filtered_prices:pd.DataFrame) -> pd.DataFrame:
        df = filtered_prices.reset_index(names='Date', drop=False).copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%Y')
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y') #Convert from string to datetime
        return df
    
    def pipeline(self) -> pd.DataFrame:
        prices = self._download_data(self.symbol, self.start_date, self.end_date, self.interval)
        filtered_prices = self._filter_cols(prices)
        df = self._reset_index(filtered_prices)
        df = self._normalize_price(df)
        df = self._log_price(df)
        df = self._pct_change(df)
        df.columns = self.col_names
        df.set_index('date', drop=True, inplace=True)
        
        return df