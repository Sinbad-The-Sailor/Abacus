# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

from src.abacus.models.model import Model


class Instrument:
    # TODO: Ensure uniqueness of identifier in whatever script creates instruments from ids.
    def __init__(self, id: int, identifier: str, instrument_type: str, price_history: pd.DataFrame):
        self.id = id
        self.identifier = identifier
        self.instrument_type = instrument_type
        self.price_history = price_history
        self._model = None

    @property
    def initial_price(self) -> float:
        return torch.Tensor(self.price_history.values.flatten())[-1]

    @property
    def log_returns(self) -> pd.DataFrame:
        # TODO: Consider which way is best to compute log returns with.
        # return np.log(self.mid_history / self.mid_history.shift(1))[1:]
        return np.log(1 + self.price_history.pct_change())[1:]

    @property
    def art_returns(self) -> pd.DataFrame:
        return np.array(self.price_history.pct_change())[1:]

    @property
    def log_returns_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.log_returns.values.flatten())

    @property
    def art_returns_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.art_returns.values.flatten())

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, other):
        self._model = other

    @property
    def price_history(self) -> pd.DataFrame:
        return self._price_history

    @price_history.setter
    def price_history(self, new):
        self._price_history = new

    def __str__(self) -> str:
        return f"{self.identifier}"

    def __repr__(self) -> str:
        return self.__str__()
