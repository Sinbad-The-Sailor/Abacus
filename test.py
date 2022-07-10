import numpy as np

from scipy.optimize import minimize
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum


class Currency(Enum):
    USD = 0,
    EUR = 1,
    GBP = 2,
    JPY = 3


class Model(ABC):
    initial_parameters: np.array
    optimal_parameters: np.array
    uniform_transformed_samples: np.array

    @abstractmethod
    def fit_model(self) -> np.array:
        pass

    @abstractmethod
    def run_simulation(self, number_of_iterations: int) -> dict:
        pass

    @abstractmethod
    def generate_uniform_samples(self):
        pass


class EquityModel(Model):

    initial_parameters = np.array([0.022, 0.3, 0.2])
    optimal_parameters = None

    def fit_model(self, data: np.array) -> bool:
        solution = minimize(
            self._cost_function, self.initial_parameters, constraints=self._constraints(), args=data)
        self.optimal_parameters = solution.x
        return solution.success

    def run_simulation(self, number_of_iterations: int) -> dict:
        print("Running equity simulation.")

    def generate_uniform_samples(self):
        print("Running equity uniforms.")

    def _cost_function(self, params: np.array, data: np.array):
        n_observations = len(data)
        log_loss = 0
        vol_est = params[0] + params[1] * \
            (data[1] ** 2) + params[2] * (data[1] ** 2)
        for i in range(2, n_observations):
            log_loss = log_loss + \
                (np.log(vol_est) + (data[i] ** 2) / vol_est)
            vol_est = params[0] + params[1] * \
                (data[i] ** 2) + params[2] * vol_est

        return log_loss

    def _constraints(self):
        constraints = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] + 1},
                       {'type': 'ineq', 'fun': lambda x:  x[0]},
                       {'type': 'ineq', 'fun': lambda x:  x[1]},
                       {'type': 'ineq', 'fun': lambda x:  x[2]}]
        return constraints


class FXModel(Model):
    def fit_model(self):
        print("Fitting FX model.")

    def run_simulation(self, number_of_iterations: int) -> dict:
        print("Running fx simulation.")

    def generate_uniform_samples(self):
        print("Running FX uniforms.")


class Instrument(ABC):
    isin: str
    local_currency: Currency
    model: Model
    price_history: dict
    return_history: dict
    has_model: bool = False

    def set_model(self, model):
        self.model = model
        self.has_model = True

    def __str__(self):
        return "Insturment hi."


class Equity(Instrument):
    ticker: str
    model: EquityModel

    def __init__(self, ticker):
        self.ticker = ticker


class FX(Instrument):
    local_currency: Currency
    foreign_currency: Currency
    model: FXModel

    def __init__(self, local_currency, foreign_currency):
        self.local_currency = local_currency
        self.foreign_currency = foreign_currency


class Portfolio():
    instruments: list
    return_distribution: list

    def __init__(self):
        pass

    def __init__(self, instruments):
        self.instruments = instruments

    def fit_models(self):

        # Check if all instruments has a model.
        self._has_models()

        # Call all fit models.
        for instrument in self.instruments:
            instrument.model.fit_model()

    def run_simulation(self) -> np.array:

        # Check if all instruments has a model.
        self._has_models()

        # Apply appropriate vine copula to all assets.

        # Run simulations.
        for instrument in self.instruments:
            instrument.model.run_simulation(1000)

    def _has_models(self):
        for instrument in self.instruments:
            if instrument.has_model == False:
                raise ValueError(f"Instrument {str(instrument)} has no model.")


def main():
    stock1 = Equity("XOM")
    stock2 = Equity("BOA")

    model1 = EquityModel()
    model2 = FXModel()

    stock1.set_model(model1)
    stock2.set_model(model1)

    fx1 = FX("USD", "EUR")
    fx2 = FX("EUR", "JPY")

    fx1.set_model(model2)
    fx2.set_model(model2)

    instruments = [stock1, stock2]

    portfolio = Portfolio(instruments=instruments)
    data = [0.01, 0.02, 0.01, -0.01]
    print(model1.optimal_parameters)


if __name__ == "__main__":
    main()
