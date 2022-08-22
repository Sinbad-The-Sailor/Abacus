# -*- coding: utf-8 -*-
import numpy as np

from abacus.instruments import Equity
from abacus.simulator.equity_models import GJRGARCHEquityModel


class ModelFactory:
    def __init__():
        # equity object property.
        # get equity -> feed into model selector -> fit model and apply it to equity. "return true".
        # More generally, we want to get insturment and feed it into different factories.
        pass

    def build_model():
        # return the equity with an added model obtained through forward CV.
        pass

    def run_model_selection():
        # run different models and check which is the best (CV).
        pass

    def equity_model_factory(equity: Equity):
        # TODO: More logic based on AIC or BIC e.g.
        initial_parametes_gjr = np.array([0.05, 0.80, 0.001])
        initial_parametes_gar = np.array([0.05, 0.80])
        model = GJRGARCHEquityModel(initial_parametes_gjr, equity.log_return_history)
        equity.set_model(model=model)
