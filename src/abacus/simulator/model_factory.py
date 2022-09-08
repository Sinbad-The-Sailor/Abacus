# -*- coding: utf-8 -*-
import numpy as np
from abacus.config import ADMISSABLE_EQUTIY_MODELS, ADMISSABLE_FX_MODELS

from abacus.instruments import FX, Equity, Instrument
from abacus.simulator.ar import AR
from abacus.simulator.garch import GARCH
from abacus.simulator.gjr_grach import GJRGARCH
from abacus.simulator.ma import MA
from abacus.simulator.model import Model
from abacus.simulator.nnar import NNAR


class ModelFactory:
    def __init__(self, instruments: list[Instrument]) -> None:
        self.instruments = instruments

    def build_model(self, data, model_name: str) -> Model:
        if model_name == "AR":
            return AR(data)
        elif model_name == "MA":
            return MA(data)
        elif model_name == "NNAR":
            return NNAR(data)
        elif model_name == "GARCH":
            return GARCH(data)
        elif model_name == "GJRGARCH":
            return GJRGARCH(data)
        else:
            raise ValueError(f"Model {model_name} not available.")

    def select_model(self, instrument: Instrument) -> None:
        current_MSE = np.Inf
        current_model = None

        if type(instrument) is Equity:
            for model_name in ADMISSABLE_EQUTIY_MODELS:
                potential_model = self.build_model(instrument.log_return_history, model_name)
                if potential_model.mse < current_MSE:
                    current_MSE = potential_model.mse
                    current_model = potential_model

        elif type(instrument) is FX:
            for model_name in ADMISSABLE_FX_MODELS:
                potential_model = self.build_model(instrument.log_return_history, model_name)
                if potential_model.mse < current_MSE:
                    current_MSE  = potential_model.mse
                    current_model = potential_model

        instrument.set_model(current_model)

    def build_all(self) -> None:
        for instrument in self.instruments:
            self.select_model(instrument)
