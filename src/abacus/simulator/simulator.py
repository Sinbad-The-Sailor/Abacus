# -*- coding: utf-8 -*-
import torch
import numpy as np
import pyvinecopulib as pv

from src.abacus.utils.instrument import Instrument
from src.abacus.utils.exceptions import ParameterError
from src.abacus.utils.enumerations import DataTypes
from src.abacus.simulator.model_selector import ModelSelector
from src.abacus.config import VINE_COPULA_FAMILIES, VINE_COPULA_NUMBER_OF_THREADS


class Simulator:
    """
    TODO: Add description of resulting tensor.

    """

    def __init__(self, instruments: list[Instrument]):
        self._model_selector = ModelSelector()
        self._instruments = sorted(instruments, key=lambda x: x.identifier)
        self._calibrated = False
        self._return_tensor = None
        self._price_tensor = None

    @property
    def covariance_matrix(self, data_type: DataTypes=DataTypes.LOG_RETURNS) -> torch.Tensor:
        # TODO: Add explanation of stacking here.
        # TODO: Ensure working under odd data length inputs.
        # TODO: Add input for price through enum.
        # TODO: Consider making static.
        # TODO: Consider check for symmetric and positive semi definiteness.
        if data_type == DataTypes.LOG_RETURNS:
            instrument_data = [instrument.log_returns_tensor for instrument in self._instruments]
        elif data_type == DataTypes.ART_RETURNS:
            instrument_data = [instrument.art_returns_tensor for instrument in self._instruments]
        else:
            raise NotImplementedError(f"Data type {data_type} not supported to build covariance matrix.")
        return torch.cov(torch.stack(instrument_data))

    @property
    def return_tensor(self) -> torch.Tensor:
        self._check_calibration()
        return self._return_tensor

    @property
    def price_tensor(self) -> torch.Tensor:
        self._check_calibration()
        if self._price_tensor is None:
            return_tensor = self.return_tensor
            inital_prices = self._inital_prices
            self._price_tensor = inital_prices * torch.exp(torch.cumsum(return_tensor, dim=1))
        return self._price_tensor

    @property
    def _uniform_samples(self) -> np.array:
        # TODO: Compute size of array and fill it vectorized. Requires a consistent number of samples accessible.
        samples = []
        for instrument in self._instruments:
            model = instrument.model
            samples.append(model.transform_to_uniform())
        return np.stack(samples).T

    @property
    def _number_of_instruments(self) -> int:
        return len(self._instruments)

    @property
    def _inital_prices(self) -> torch.Tensor:
        size = (self._number_of_instruments, )
        intial_prices = torch.empty(size=size)
        for i, instrument in enumerate(self._instruments):
            intial_prices[i] = instrument.initial_price
        return intial_prices[:, None, None]

    def calibrate(self):
        self._calibrate_instruments()
        self._calibrate_copula()
        self._calibrated = True

    def run_simulation(self, time_steps: int, number_of_simulations: int) -> torch.Tensor:
        assert isinstance(time_steps, int) and time_steps > 0, "Time step must be a positive integer."
        assert isinstance(number_of_simulations, int) and number_of_simulations > 0, "Number of simulations must be a positive integer."
        number_of_instruments = self._number_of_instruments
        size = (number_of_instruments, time_steps, number_of_simulations)
        simulation_tensor = torch.empty(size=size)

        for n in range(number_of_simulations):
            simulations = self._coupla.simulate(time_steps).T
            for i, simulation in enumerate(simulations):
                simulation_tensor[i,:,n] = self._instruments[i].model.transform_to_true(torch.tensor(simulation))
        self._return_tensor = simulation_tensor

    def _calibrate_instruments(self):
        for instrument in self._instruments:
            self._model_selector.instrument = instrument
            self._model_selector.select_model()

    def _calibrate_copula(self):
        uniforms = self._uniform_samples
        controls = pv.FitControlsVinecop(family_set=VINE_COPULA_FAMILIES, num_threads=VINE_COPULA_NUMBER_OF_THREADS)
        copula = pv.Vinecop(uniforms, controls=controls)
        self._coupla = copula

    def _check_calibration(self):
        if not self._calibrated:
            raise ParameterError("Simulator is not calibrated.")
