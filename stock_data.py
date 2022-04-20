import pandas as pd
import numpy as np


class StockData:
    # List of data in ascending order! Since dicts are hash-tables one might think about using dicts with
    # dates as a key

    def __init__(self, ric):
        self.adj_close = pd.DataFrame()
        self.ric = ric

    def get_log_returns(self):
        """
        Calculated logarithmic returns of adjusted close prices.

        Returns: List of adjusted close logarithmic returns.
        """

        return np.log(self.adj_close / self.adj_close.shift(1))[1:]