# -*- coding: utf-8 -*-
import numpy as np
from abacus.config import ADMISSIBLE_EQUTIY_MODELS, ADMISSIBLE_FX_MODELS

from abacus.instruments import FX, Equity, Instrument
from abacus.simulator.ar import AR
from abacus.simulator.garch import GARCH
from abacus.simulator.gjr_grach import GJRGARCH
from abacus.simulator.ma import MA
from abacus.simulator.model import Model
from abacus.simulator.nnar import NNAR


class ModelFactory:
    """
    Model factory for instruments. Picks the most appropriate model for each
    instrument using an error estimate.

    Currently using MSE (Mean Squared Error).
    """

    def __init__(self, instruments: list[Instrument]) -> None:
        self.instruments = instruments

    @staticmethod
    def build_model(data, model_name, *args) -> Model:
        if model_name == "AR":
            return [AR(data, p=arg) for arg in args]
        elif model_name == "MA":
            return [MA(data, q=arg) for arg in args]
        elif model_name == "NNAR":
            return [NNAR(data, p=arg) for arg in args]
        elif model_name == "GARCH":
            return [GARCH(data)]
        elif model_name == "GJRGARCH":
            return [GJRGARCH(data)]
        else:
            raise ValueError(f"Model {model_name} not available.")

    def select_model(self, instrument: Instrument) -> None:
        current_MSE = np.Inf
        current_model = None

        if type(instrument) is Equity:
            for model_name, hyperparameters in ADMISSIBLE_EQUTIY_MODELS.items():
                potential_models = self.build_model(
                    np.array(instrument.log_return_history),
                    model_name,
                    *hyperparameters,
                )
                for potential_model in potential_models:
                    potential_model.fit_model()
                    if potential_model.mse < current_MSE:
                        current_MSE = potential_model.mse
                        current_model = potential_model

        elif type(instrument) is FX:
            for model_name, hyperparameters in ADMISSIBLE_EQUTIY_MODELS:
                potential_models = self.build_model(
                    np.array(instrument.log_return_history),
                    model_name,
                    *hyperparameters,
                )
                for potential_model in potential_models:
                    potential_model.fit_model()
                    if potential_model.mse < current_MSE:
                        current_MSE = potential_model.mse
                        current_model = potential_model

        instrument.set_model(current_model)

    def build_all(self) -> None:
        """
        Applies a model builder for every instrument given.
        """
        for instrument in self.instruments:
            self.select_model(instrument)
