# -*- coding: utf-8 -*-
import numpy as np

from pandas_datareader import data as pdr

from config import ABACUS_DATABASE_CONNECTION
from database.database_parser import select_price_data
from sqlite3 import Error


class StockData:
    """
    Represents an equity object. Contains the necessary information for equity modelling fitting. Note, this object is
    currently used as an asset object, meaning the use is not restricted to equities.
    """

    def __init__(self, ric):
        self.ric = ric
        try:
            self.adj_close = select_price_data(ABACUS_DATABASE_CONNECTION, ric)
        except Error as e:
            print(e)

    def get_log_returns(self):
        """
        Calculates logarithmic returns of adjusted close prices.

        Returns: List of adjusted close logarithmic returns.
        """

        return np.log(self.adj_close / self.adj_close.shift(1))[1:]

    def parse_yahoo_data(self):
        """
        Fetches close prices from Yahoo Finance with between predetermined times with
        weekly intervals.
        """

        data = pdr.get_data_yahoo(
            self.ric, start="2011-12-28", end="2021-12-28", interval="wk")
        self.adj_close = data['Adj Close']
