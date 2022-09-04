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
        """
        Runs univariate simulation of process.

        Args:
            number_of_steps (int): number of simulation steps into the future.

        Returns:
            np.array: simulated process.
        """
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

    def transform_to_true(self, uniform_sample: np.array) -> np.array:
        """
        Transforms a predicted uniform sample to true values of the process. Very similar to the
        univarite simulation case, the difference is only that uniform samples are obtained from
        elsewhere.

        Args:
            uniform_sample (np.array): sample of uniform variables U(0,1).

        Returns:
            np.array: simulated process.
        """
        number_of_observations = len(uniform_sample)
        simulated_process = np.zeros(number_of_observations)
        current_regression_values = self.data[-self.p:]
        mu = self.solution[0]
        sigma = self.solution[1]
        phi = self.solution[2:]

        for i in range(number_of_observations):
            residual = norm.ppf(uniform_sample[i])
            simulated_process[i] = mu + phi.T @ current_regression_values + sigma * residual
            current_regression_values = np.insert(current_regression_values[:-1], 0, simulated_process[i])

        return simulated_process

    def transform_to_uniform(self) -> np.array:
        """
        Transformes the normalized time series to uniform variables, assuming Gaussian White Noise. Uses
        a standard normalization approach for the first p values to avoid shrinking the dataset.

        Returns:
            np.array: sample of uniform variables U(0,1).
        """
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

    def _characteristic_roots(self, coefficients: np.array) -> np.array:
        """
        Calculates the roots of the AR(p) characteristic polynomial to check for non-weak-stationarity.

        Args:
            coefficients (np.array): coefficients of the polynomial.

        Returns:
            np.array: roots of the polynomial.
        """
        pass

    def _check_unit_roots(self) -> None:
        """
        Checks for unit roots outside the unit circle.

        Raises:
            ValueError: raised if unit root is found.
        """
        raise ValueError("Stationarity encountered.")
