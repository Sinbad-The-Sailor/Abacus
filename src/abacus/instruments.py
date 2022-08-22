# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from datetime import datetime
from abc import ABC
from abacus.simulator.model import Model
from abacus.simulator.equity_models import EquityModel
from abacus.simulator.fx_models import FXModel
from abacus.utilities.currency_enum import Currency
from pandas_datareader import data as pdr


class Instrument(ABC):
    ric: str
    local_currency: Currency
    start_date: datetime
    end_date: datetime
    interval: str
    price_history: pd.DataFrame
    return_history: pd.DataFrame
    model: Model
    has_model: bool = False

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

        try:
            self.price_history = pdr.get_data_yahoo(
                ric, start=start_date, end=end_date, interval=interval
            )["Adj Close"]
            self.art_return_history = self._art_return_history(
                price_history=self.price_history
            )
            self.log_return_history = self._log_return_history(
                price_history=self.price_history
            )
        except:
            # TODO: Add backup fetching and custom error.
            print("Cannot fetch yahoo data.")

    def set_model(self, model: Model) -> None:
        self.model = model
        self.has_model = True

    def art_return_history():
        pass

    def log_return_history():
        pass

    def _establish_connection(self):
        pass

    def _has_prices(self):
        if self.price_hisotry is None:
            return False
        return True

    @staticmethod
    def _art_return_history(price_history: pd.DataFrame) -> pd.DataFrame:
        return price_history / price_history.shift(1)[1:]

    @staticmethod
    def _log_return_history(price_history: pd.DataFrame) -> pd.DataFrame:
        return np.log(price_history / price_history.shift(1))[1:]

    def __str__(self) -> str:
        return "instrument"


class Equity(Instrument):
    model: EquityModel

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
    model: FXModel

    def __init__(
        self,
        ric: str,
        currency: Currency,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ):
        super().__init__(ric, currency, start_date, end_date, interval)
