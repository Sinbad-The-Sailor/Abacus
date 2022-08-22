# -*- coding: utf-8 -*-
import numpy as np
import copulae as cop
import pyvinecopulib as pv
from abacus.abacus_simulator.model_factory import ModelFactory

from abacus.instruments import Instrument
from abacus.config import DEFALUT_STEPS, VINE_COPULA_FAMILIES, DEFALUT_SIMULATIONS


class Simulator:

    def __init__(self, instruments: list[Instrument]):
        self.instruments = instruments
        self.model_factory = ModelFactory

        try:
            self.number_of_instruments = len(instruments)
        except ValueError:
            # log that no instruments are inserted. Critical.
            self.number_of_instruments = -1

        try:
            last_prices = self._last_prices()
        except ValueError:
            # log that no prices are available. Critical.
            self.value = -1

        self.find_models()
        self.fit_models()
        self.fit_portfolio()

    def find_models(self):
        for instrument in self.instruments:
            self.model_factory.equity_model_factory(instrument)

    def fit_models(self):
        # Check if all instruments has a model.
        if not self._has_models():
            raise ValueError(f"One instrument has no model.")

        # Call all fit models.
        for instrument in self.instruments:
            instrument.model.fit_model()

    def fit_portfolio(self):

        # Check if more than 1 asset exists.
        if self.number_of_instruments == 1:
            raise ValueError("To few instruments to run dependency.")

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

        if self.number_of_instruments > 2:
            # Function which picks optimal vine copula.
            controls = pv.FitControlsVinecop(family_set=VINE_COPULA_FAMILIES)
            copula = pv.Vinecop(uniforms, controls=controls)

        self.copula = copula

    def run_simultion_assets(
        self, number_of_steps: int = DEFALUT_STEPS, dependency: bool = True
    ) -> np.array:

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
            return self._generate_multivariate_simulation(
                number_of_steps=number_of_steps
            )
        else:
            return self._generate_univariate_simulation(number_of_steps=number_of_steps)

    def run_simulation_portfolio(
        self,
        number_of_steps: int = DEFALUT_STEPS,
        number_of_simulations: int = DEFALUT_SIMULATIONS,
        dependency: bool = True,
    ) -> list[np.array]:

        init_prices = self._last_prices()

        for simulation in range(number_of_simulations):
            # Portfolio constituance simulation.
            simultion_matrix = self.run_simultion_assets(
                number_of_steps=number_of_steps, dependency=dependency
            )

            # Portfolio prices.
            temp_prices = init_prices * \
                np.prod(np.exp(simultion_matrix), axis=1)

            print(simultion_matrix)
            print(simulation)

        return temp_prices

    def _generate_univariate_simulation(self, number_of_steps: int) -> np.array:
        # TODO: Remove list to make this faster!
        result = []

        for instrument in self.instruments:
            result.append(
                instrument.model.run_simulation(
                    number_of_steps=number_of_steps)
            )

        return np.vstack(result)

    def _generate_multivariate_simulation(self, number_of_steps: int) -> np.array:

        # Check if copula has been fitted.
        if not self._has_copula():
            raise ValueError("Portfolio has no multivarite model/copula.")

        if self.number_of_instruments == 2:
            simulated_uniforms = self.copula.random(number_of_steps)

        if self.number_of_instruments > 2:
            simulated_uniforms = self.copula.simulate(number_of_steps)

        # TODO: Remove list to make this faster if need be.
        result = []
        for i in range(self.number_of_instruments):
            current_instrument = self.instruments[i]
            current_uniform_sample = simulated_uniforms[:, i]
            result.append(
                current_instrument.model.generate_correct_samples(
                    current_uniform_sample
                )
            )

        return result

    def _last_prices(self) -> np.array:
        try:
            # TODO: Assert that all dates are the same!
            last_prices = np.zeros(self.number_of_instruments)
            for i in range(self.number_of_instruments):
                last_prices[i] = self.instruments[i].price_history[-1]
            return last_prices
        except ValueError:
            self._last_prices = np.zeros(self.number_of_instruments)

    def _has_models(self) -> bool:
        for instrument in self.instruments:
            if instrument.has_model == False:
                return False
        return True

    def _has_instruments(self) -> bool:
        if self.number_of_instruments == 0:
            return False
        return True

    def _has_copula(self) -> bool:
        if self.copula is None:
            return False
        return True

    def _has_solution(self) -> bool:
        for instrument in self.instruments:
            if instrument.model._has_solution == False:
                return False
        return True

    def __len__(self):
        return self.number_of_instruments

    def __str__(self):
        raise NotImplemented

    def __repr__(self):
        raise NotImplemented
