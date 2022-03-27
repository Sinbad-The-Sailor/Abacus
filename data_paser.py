# REMOVE pandas_datareader if Refinitv is used later.

from pandas_datareader import data as pdr
from equity_model import StockData


def parse_yahoo_data(asset: StockData) -> StockData:
    data = pdr.get_data_yahoo(asset.ric, start="2011-12-28", end="2021-12-28", interval="wk")
    asset.adj_close = data['Adj Close']
