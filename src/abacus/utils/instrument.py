# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd



class Instrument:
    def __init__(self, identifier: str, instrument_type: str, price_history: pd.DataFrame):
        self.identifier = identifier
        self.instrument_type = instrument_type
        self.price_history = price_history
        self._model = None

    @property
    def initial_price(self) -> float:
        return torch.Tensor(self.price_history.values.flatten())[-1]

    @property
    def log_returns(self) -> pd.DataFrame:
        # return np.log(self.mid_history / self.mid_history.shift(1))[1:]
        return np.log(1 + self.price_history.pct_change())[1:]

    @property
    def art_returns(self) -> pd.DataFrame:
        return self.price_history.pct_change()[1:]

    @property
    def log_returns_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.log_returns.values.flatten())

    @property
    def art_returns_tensor(self) -> torch.Tensor:
        return torch.Tensor(self.art_returns.values.flatten())

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, other):
        self._model = other
