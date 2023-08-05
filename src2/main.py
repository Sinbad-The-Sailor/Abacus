# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np

from datetime import datetime
from matplotlib import pyplot as plt

from utils.stock_factory import StockFactory
from utils.instruments import Stock
from utils.config import STOCK_ADMISSIBLE_MODELS

from models.ar import AR
from models.garch import GARCH



start = datetime.strptime("2005-05-01", r"%Y-%m-%d")
end = datetime.strptime("2023-06-01", r"%Y-%m-%d")
instrument_specification = ["XOM", "GS", "T"]
instrument_factory = StockFactory(tickers=instrument_specification,
                                  start=start,
                                  end=end)
stocks = instrument_factory.build_stocks()



class Simulator:

    def __init__(self, instruments: list):
        self._instruments = instruments
        self._calibrated = False

    def calibrate(self):
        self._calibrate_instruments()
        self._calibrate_copula()
        self._calibrated = True

    def _calibrate_instruments(self):
        for instrument in self._instruments:
            if isinstance(instrument, Stock):
                self._calibrate_stock(instrument)

    def _calibrate_copula(self):
        ...

    def _calibrate_stock(self, stock):
        current_aic = np.inf
        risk_factor = stock.risk_factors[0]
        data = risk_factor.price_history.log_returns

        for model_name in STOCK_ADMISSIBLE_MODELS:
            if model_name == "AR":
                model = AR(data)
                model.calibrate()
                if model.aic < current_aic:
                    risk_factor.model = model

            elif model_name == "GARCH":
                model = GARCH(data)
                model.calibrate()
                if model.aic < current_aic:
                    risk_factor.model = model


    def run_simulation(time_steps: int) -> torch.Tensor:
        # Check for succesful calibration, Throw an error otherwise.
        ...
