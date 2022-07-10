import numpy as np
import pandas as pd

from datetime import datetime
from abc import ABC
from test_models import Model, EquityModel, FXModel
from test_enums import Currency
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

    def __init__(self, ric: str, currency: Currency, start_date: datetime, end_date: datetime, interval: str):
        self.ric = ric
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

        try:
            price_history = pdr.get_data_yahoo(
                ric, start=start_date, end=end_date, interval=interval)['Adj Close']
            return_history = self._log_return_history(
                price_history=price_history)

            self.price_history = price_history
            self.return_history = return_history

        except:
            # TODO: Add backup fetching and custom error.
            print("Cannot fetch yahoo data.")

    def set_model(self, model: Model) -> None:
        self.model = model
        self.has_model = True

    @staticmethod
    def _log_return_history(price_history: pd.DataFrame) -> pd.DataFrame:
        return np.log(price_history / price_history.shift(1))[1:]

    def __str__(self) -> str:
        return "instrument"


class Equity(Instrument):
    model: EquityModel

    def __init__(self, ric: str, currency: Currency, start_date: datetime, end_date: datetime, interval: str):
        super().__init__(ric, currency, start_date, end_date, interval)


class FX(Instrument):
    model: FXModel

    def __init__(self, ric: str, currency: Currency, start_date: datetime, end_date: datetime, interval: str):
        super().__init__(ric, currency, start_date, end_date, interval)
