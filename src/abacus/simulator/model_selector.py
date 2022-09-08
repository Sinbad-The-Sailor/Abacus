# -*- coding: utf-8 -*-
import numpy as np

from abacus.instruments import Equity, Instrument, FX
from abacus.simulator.garch import GARCH
from abacus.simulator.gjr_grach import GJRGARCH


class ModelSelector:
    """
    Model factory for instruments. Picks the most appropriate model for each
    instrument using an error estimate.

    Currently using MSE (Mean Squared Error).
    """

    def __init__(self, instruments: list[Instrument]):
        self.instruments = instruments

    def build_all(self):
        """
        Applies a model builder for every instrument given.
        """
        for instrument in self.instruments:
            self.build_model(instrument)

    def build_model(self, instrument: Instrument):
        """
        Matches type of instrument with appropriate builder.

        Args:
            instrument (Instrument): target instrument
        """
        if isinstance(instrument, Equity):
            self._build_equity(instrument)
        elif isinstance(instrument, FX):
            self._build_FX(instrument)

























    def _build_equity(self, instrument: Equity):
        """
        Finds and builds an Equity model for a Equity instrument. Can only use
        Equity admissable models.

        Args:
            instrument (Equity): input Equity instrument.
        """
        minimal_aic = np.Inf
        minimal_model = None

        gjr_model = GJRGARCHModel(data=instrument.log_return_history)
        gjr_model.fit_model()
        if gjr_model.aic() < minimal_aic:
            minimal_aic = gjr_model.aic()
            minimal_model = gjr_model
        gar_model = GARCHModel(data=instrument.log_return_history)
        gar_model.fit_model()
        if gar_model.aic() < minimal_aic:
            minimal_aic = gar_model.aic()
            minimal_model = gar_model

        instrument.set_model(minimal_model)

    def _build_FX(self, instrument: FX):
        """
        Finds and builds an FX model for a FX instrument. Can only use
        FX admissable models.

        Args:
            instrument (FX): input FX instrument.
        """
        minimal_aic = np.Inf
        minimal_model = None

        gjr_model = GJRGARCHModel(data=instrument.log_return_history)
        gjr_model.fit_model()
        if gjr_model.aic() < minimal_aic:
            minimal_aic = gjr_model.aic()
            minimal_model = gjr_model
        gar_model = GARCHModel(data=instrument.log_return_history)
        gar_model.fit_model()
        if gar_model.aic() < minimal_aic:
            minimal_aic = gar_model.aic()
            minimal_model = gar_model

        instrument.set_model(minimal_model)
