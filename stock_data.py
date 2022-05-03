import pandas as pd
import numpy as np

from config import ABACUS_DATABASE_CONNECTION
from pandas_datareader import data as pdr
from database.database_parser import select_price_data


class StockData:

    def __init__(self, ric):
        self.adj_close = select_price_data(ABACUS_DATABASE_CONNECTION, ric)
        self.ric = ric

    def get_log_returns(self):
        """
        Calculated logarithmic returns of adjusted close prices.

        Returns: List of adjusted close logarithmic returns.
        """

        return np.log(self.adj_close / self.adj_close.shift(1))[1:]

    def parse_yahoo_data(self):
        data = pdr.get_data_yahoo(self.ric, start="2011-12-28", end="2021-12-28", interval="wk")
        self.adj_close = data['Adj Close']
