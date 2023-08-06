# -*- coding: utf-8 -*-

import torch
import numpy as np
import pyvinecopulib as pv

from utils.config import VINE_COPULA_FAMILIES, VINE_COPULA_NUMBER_OF_THREADS, STOCK_ADMISSIBLE_MODELS
from utils.instruments import Stock, RiskFactor
from models.ar import AR
from models.garch import GARCH

class Simulator:

    def __init__(self, instruments: list):
        self._instruments = instruments
        self._calibrated = False

    @property
    def _uniform_samples(self) -> np.array:
        samples = []
        for instrument in self._instruments:
            risk_factor = instrument.risk_factors[0]
            model = risk_factor.model
            samples.append(model.transform_to_uniform())
        return np.stack(samples).T

    @property
    def _number_of_risk_factors(self) -> int:
        return len(self._risk_factors)

    @property
    def _risk_factors(self) -> list[RiskFactor]:
        risk_factors = []
        for instrument in self._instruments:
            risk_factors += instrument.risk_factors
        return risk_factors

    def calibrate(self):
        self._calibrate_instruments()
        self._calibrate_copula()
        self._calibrated = True

    def run_simulation(self, time_steps: int, number_of_simulations: int) -> torch.Tensor:
        number_of_risk_factors = self._number_of_risk_factors
        size = (number_of_risk_factors, time_steps, number_of_simulations)
        simulation_tensor = torch.empty(size=size)

        print("number of simulations", number_of_simulations)
        print("time steps", time_steps)
        print("number of risk factors", self._number_of_risk_factors)
        print("number of simulations from cop", len(self._coupla.simulate(time_steps)))

        for n in range(number_of_simulations):
            simulations = self._coupla.simulate(time_steps).T
            for i, simulation in enumerate(simulations):
                simulation_tensor[i,:,n] = self._risk_factors[i].model.transform_to_true(torch.tensor(simulation))
        return simulation_tensor

    def _calibrate_instruments(self):
        for instrument in self._instruments:
            if isinstance(instrument, Stock):
                self._calibrate_stock(instrument)

    def _calibrate_copula(self):
        uniforms = self._uniform_samples
        controls = pv.FitControlsVinecop(family_set=VINE_COPULA_FAMILIES,
                                         num_threads=VINE_COPULA_NUMBER_OF_THREADS)
        copula = pv.Vinecop(uniforms, controls=controls)
        self._coupla = copula

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
