# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from datetime import datetime
from abc import ABC
from abacus.simulator.model import Model
from abacus.utilities.currency_enum import Currency


class Instrument(ABC):
    def __init__(
        self,
        ric: str,
        currency: Currency,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ):
        self.ric = ric
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.local_currency = currency

        self.price_history = None
        self.art_return_history = None
        self.log_return_history = None
        self.model = None
        self.has_model = False

    def set_model(self, model: Model):
        self.model = model
        self.has_model = True

    def set_price_history(self, price_history: pd.DataFrame):
        self.price_history = price_history
        self.art_return_history = self._art_return_history()
        self.log_return_history = self._log_return_history()

    def _art_return_history(self) -> pd.DataFrame:
        if self._has_prices:
            return self.price_history / self.price_history.shift(1)[1:]
        else:
            raise ValueError("No price history exists.")

    def _log_return_history(self) -> pd.DataFrame:
        if self._has_prices:
            return np.log(self.price_history / self.price_history.shift(1))[1:]
        else:
            raise ValueError("No price history exists.")

    def _has_prices(self) -> bool:
        """
        Checks if instrument has price history dataframe.

        Returns:
            bool: status of price history.
        """
        if self.price_hisotry is None:
            return False
        return True


class Equity(Instrument):
    def __init__(
        self,
        ric: str,
        currency: Currency,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ):
        super().__init__(ric, currency, start_date, end_date, interval)


class FX(Instrument):
    def __init__(
        self,
        ric: str,
        currency: Currency,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ):
        super().__init__(ric, currency, start_date, end_date, interval)
