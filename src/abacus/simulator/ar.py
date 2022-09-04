# -*- coding: utf-8 -*-
import numpy as np

from numpy.linalg import inv
from scipy.stats import norm
from abacus.simulator.model_new import Model


class AR(Model):
    def __init__(self, data, p):
        super().__init__(data)
        self.p = p

    @property
    def initial_solution(self) -> np.array:
        """
        Not required for AR(p) model.
        """
        pass

    @property
    def mse(self) -> float:
        pass

    def fit_model(self) -> np.array:
        """
        Fits parameters of the AR(p) model by solving the normal equations.

        Returns:
            np.array: optimal parameters.
        """
        lag = self.p
        number_of_observations = len(self.data)

        y = self.data[lag:]
        x = []
        for i in range(lag):
            x.append(self.data[lag-i-1:number_of_observations-i-1])
        x = np.stack(x).T
        phi = inv(x.T @ x) @ x.T @ y

        parameters = np.zeros(self.p+2)
        parameters[0] = np.mean(self.data)
        parameters[1] = np.std(self.data)
        parameters[2:] = phi
        self.solution = parameters

        return parameters

    def _cost_function(self) -> float:
        """
        Not required for AR(p) model.
        """
        pass

    def run_simulation(self, number_of_steps: int) -> np.array:
        simulated_process = np.zeros(number_of_steps)
        current_regression_values = self.data[-self.p:]
        mu = self.solution[0]
        sigma = self.solution[1]
        phi = self.solution[2:]

        for i in range(number_of_steps):
            residual = np.random.normal()
            simulated_process[i] = mu + phi.T @ current_regression_values + sigma * residual
            current_regression_values = np.insert(current_regression_values[:-1], 0, simulated_process[i])

        return simulated_process

    def transform_to_true(self) -> np.array:
        pass

    def transform_to_uniform(self) -> np.array:
        number_of_observations = len(self.data)
        uniform_sample = np.zeros(number_of_observations)
        current_regression_values = self.data[:self.p]
        mu = self.solution[0]
        sigma = self.solution[1]
        phi = self.solution[2:]

        for i in range(number_of_observations):
            if i <= self.p-1:
                uniform_sample[i] = norm.cdf((self.data[i]-mu)/sigma)
            else:
                uniform_sample[i] = norm.cdf((self.data[i] - mu - phi.T @ current_regression_values)/sigma)
                current_regression_values = np.insert(current_regression_values[:-1], 0, self.data[i])

        return uniform_sample

    def _characteristic_roots(self, solution: np.array) -> np.array:
        pass

    def _check_unit_roots(self, solution: np.array) -> None:
        pass
