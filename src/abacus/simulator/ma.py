# -*- coding: utf-8 -*-
import logging
import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm
from abacus.simulator.model_new import Model
from abacus.config import EPSILON


logger = logging.getLogger(__name__)


class MA(Model):
    def __init__(self, data: np.array, q: int):
        super().__init__(data)
        self.q = q

    @property
    def inital_solution(self) -> np.array:
        """
        Common sense intial values for each parameter.

        Returns:
            np.array: list of inital values for parameters. Formatted as: [mu, sigma, theta_1, ... , theta_q].
        """
        result = np.ones(self.q + 2) * 0.05
        mu = np.mean(self.data)
        sigma = np.std(self.data)
        result[0] = mu
        result[1] = sigma
        return result

    @property
    def mse(self) -> float:
        """
        Calculates the mean squared error.

        Returns:
            float: sum of mean squared errors.
        """
        return np.sum(self._generate_residuals(self.solution) ** 2)

    def fit_model(self) -> np.array:
        """
        Fits model with trust-constr method. Nelder-Mead is also suitable for this model.

        Returns:
            np.array: optimal parameters.
        """
        inital_solution = self.inital_solution
        solution = minimize(
            fun=self._cost_function, x0=inital_solution, method="trust-constr"
        )
        if not solution.success:
            logger.error(f"optimizer success {solution.success}")
        self.solution = solution.x
        return solution.x

    def _cost_function(self, params: np.array) -> float:
        """
        Defines the conditional log loss for the MA model. Calculates the loss recursively.

        Args:
            params (np.array): The first element is the sigma paramter. The rest are the theta parameters.

        Returns:
            float: log loss value.
        """
        number_of_observations = len(self.data)
        residuals = self._generate_residuals(params)
        sigma = params[1]
        return np.sum(((residuals) / sigma) ** 2) + number_of_observations * np.log(
            sigma ** 2 + EPSILON
        )

    def run_simulation(self, number_of_steps: int) -> np.array:
        """
        Runs univariate simulation of process.

        Args:
            number_of_steps (int): number of simulation steps into the future.

        Returns:
            np.array: simulated process.
        """
        simulated_process = np.zeros(number_of_steps)
        current_residuals = np.flip(self._generate_residuals(self.solution)[-self.q :])
        mu = self.solution[0]
        sigma = self.solution[1]
        theta = self.solution[2:]

        for i in range(number_of_steps):
            residual = np.random.normal()
            simulated_process[i] = mu + theta.T @ current_residuals + sigma * residual
            current_residuals = np.insert(current_residuals[:-1], 0, sigma * residual)

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
        current_residuals = np.flip(self._generate_residuals(self.solution)[-self.q :])
        mu = self.solution[0]
        sigma = self.solution[1]
        theta = self.solution[2:]

        for i in range(0, number_of_observations):
            residual = norm.ppf(uniform_sample[i])
            simulated_process[i] = mu + theta.T @ current_residuals + sigma * residual
            current_residuals = np.insert(current_residuals[:-1], 0, sigma * residual)

        return simulated_process

    def transform_to_uniform(self) -> np.array:
        """
        Transformes the normalized time series to uniform variables, assuming Gaussian White Noise.

        Returns:
            np.array: sample of uniform variables U(0,1).
        """
        number_of_observations = len(self.data)
        uniform_sample = np.zeros(number_of_observations)
        sigma = self.solution[1]
        residuals = self._generate_residuals(self.solution)

        for i in range(number_of_observations):
            uniform_sample[i] = norm.cdf(residuals[i] / sigma)

        return uniform_sample

    def _generate_residuals(self, params: np.array) -> np.array:
        """
        Helper method to recursivley generate residuals based on some set of values for params.

        Args:
            params (np.array): parameters of the model.

        Returns:
            np.array: residuals calculated based of the guessed parameters.
        """
        number_of_observations = len(self.data)
        residuals = np.zeros(number_of_observations)
        z = np.zeros(self.q)

        mu = params[0]
        theta = params[2:]

        for i in range(0, number_of_observations):
            residual = self.data[i] - (mu + theta.T @ z)
            residuals[i] = residual
            z = np.insert(z[:-1], 0, residual)

        return residuals
