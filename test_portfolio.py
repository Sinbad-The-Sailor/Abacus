import numpy as np
import pandas as pd

import copulae as cop
import pyvinecopulib as pv

from test_instruments import Instrument
from test_config import VINE_COPULA_FAMILIES

SIMULATIONS = 1000


class Portfolio():

    def __init__(self, instruments: list[Instrument], init_value: float, init_positions: list[int]):
        self.instruments = instruments
        self.init_value = init_value
        self.init_positions = init_positions
        self.return_distribution = None
        try:
            self.number_of_instruments = len(instruments)
        except ValueError:
            self.number_of_instruments = 0

    def fit_models(self):

        # Check if all instruments has a model.
        if not self._has_models():
            raise ValueError(f"One instrument has no model.")

        # Call all fit models.
        for instrument in self.instruments:
            instrument.model.fit_model()

    def run_simulation(self, number_of_iterations: int = SIMULATIONS, dependency: bool = True) -> np.array:

        # Check if portfolio has instruments.
        if not self._has_instruments():
            raise ValueError("Portfolio has no instruments.")

        # Check if all instruments has a model.
        if not self._has_models():
            raise ValueError("One instrument has no model.")

        # Check if all models are fitted.
        if not self._has_solution():
            raise ValueError("One model has no solution.")

        if dependency:
            return self._generate_multivariate_simulation(number_of_iterations=number_of_iterations)
        else:
            return self._generate_univariate_simulation(number_of_iterations=number_of_iterations)

    def _generate_univariate_simulation(self, number_of_iterations: int) -> np.array:
        # TODO: Remove list to make this faster!
        result = []

        for instrument in self.instruments:
            result.append(instrument.model.run_simulation(
                number_of_iterations=number_of_iterations))

        return np.vstack(result)

    def _generate_multivariate_simulation(self, number_of_iterations: int) -> np.array:

        # Check if more than 1 asset exists.
        if self.number_of_instruments == 1:
            raise ValueError("To few instruments to run dependency.")

        # Creating uniform data.
        # TODO: Remove list to make this faster!
        # TODO: Assert lenght -> pick smallest if error.
        # TODO: Create dict for insturments in portfolio in order to never mix up returns!
        uniforms = []
        for instrument in self.instruments:
            uniforms.append(instrument.model.generate_uniform_samples())
        uniforms = np.stack(uniforms).T

        if self.number_of_instruments == 2:
            # TODO: Function which picks optimal vanilla copula from a family of copulae.
            copula = cop.StudentCopula()
            copula.fit(uniforms)
            simulated_uniforms = copula.random(number_of_iterations)

        if self.number_of_instruments > 2:
            # Function which picks optimal vine copula.
            controls = pv.FitControlsVinecop(family_set=VINE_COPULA_FAMILIES)
            copula = pv.Vinecop(uniforms, controls=controls)
            simulated_uniforms = copula.simulate(number_of_iterations)
            print(copula)

        result = []
        for i in range(0, self.number_of_instruments):
            current_instrument = self.instruments[i]
            current_uniform_sample = simulated_uniforms[:, i]
            result.append(
                current_instrument.model.generate_correct_samples(current_uniform_sample))

        return result

    def _has_models(self) -> bool:
        for instrument in self.instruments:
            if instrument.has_model == False:
                return False
        return True

    def _has_instruments(self) -> bool:
        if self.number_of_instruments == 0:
            return False
        return True

    def _has_solution(self) -> bool:
        for instrument in self.instruments:
            if instrument.model._has_solution == False:
                return False
        return True

    def __len__(self):
        return self.number_of_instruments
