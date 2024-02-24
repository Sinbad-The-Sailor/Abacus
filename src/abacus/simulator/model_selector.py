# -*- coding: utf-8 -*-
import logging

import numpy as np

from src.abacus.utils.instrument import Instrument
from src.abacus.models.ar import AR
from src.abacus.models.garch import GARCH


logger = logging.getLogger(__name__)



class ModelSelector:

    STOCK_ADMISSIBLE_MODELS = AR, GARCH

    def __init__(self):
        self._instrument = None

    @property
    def instrument(self) -> Instrument | None:
        return self._instrument

    @instrument.setter
    def instrument(self, other: Instrument):
        self._instrument = other

    def select_model(self):
        instrument_type = self.instrument.instrument_type
        if instrument_type == "Stock":
            self._select_stock_model()
        else:
            raise NotImplementedError(f"Instrument type {instrument_type} of {self.instrument} is not recognized.")

    def _select_stock_model(self):
        current_aic = np.inf
        data = self.instrument.log_returns_tensor
        for Model in self.STOCK_ADMISSIBLE_MODELS:
            model = Model(data=data)
            model.calibrate()
            if model.aic < current_aic:
                best_model = model
        self.instrument.model = best_model
        logging.debug(f"{best_model} was selected for {self.instrument}.")
