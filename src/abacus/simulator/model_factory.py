# -*- coding: utf-8 -*-
import numpy as np

from abacus.instruments import Equity, Instrument, FX
from abacus.simulator.equity_models import GJRGARCHEquityModel


class ModelFactory:
    def __init__():
        pass

    def build_model():
        pass

    def run_model_selection():
        # run different models and check which is the best (CV).
        pass

    def equity_model_factory(equity: Equity):
        initial_parametes_gjr = np.array([0.05, 0.80, 0.001])
        initial_parametes_gar = np.array([0.05, 0.80])
        model = GJRGARCHEquityModel(initial_parametes_gjr, equity.log_return_history)
        equity.set_model(model=model)


class ModelSelector:
    """
    Model factory for instruments. Picks the most appropriate model for each
    instrument using information criteria.

    Currently using AIC (Akaike Information Criterion).
    """

    ELIGABLE_EQUITY_MODELS = []
    ELIGABLE_FX_MODELS = []

    def __init__(self, instruments: list[Instrument]):
        self.instruments = instruments

    def build_model(self, instrument):
        if isinstance(instrument, Equity):
            self._build_equity(instrument)
        elif isinstance(instrument, FX):
            self._build_FX(instrument)

    def build_all(self):
        """
        Applies a model builder for every instrument given.
        """
        for instrument in self.instrument:
            self.build_model(instrument)

    def _build_equity(self):
        minimal_aic = None
        for model in self.ELIGABLE_EQUITY_MODELS:
            pass

    def _build_FX(self):
        minimal_aic = None
        for model in self.ELIGABLE_FX_MODELS:
            pass

    def aic(self, likelihood, parameters):
        pass
