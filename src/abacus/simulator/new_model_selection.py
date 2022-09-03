# -*- coding: utf-8 -*-

import numpy as np

from abc import ABC, abstractmethod
from scipy.optimize import minimize

import logging
logger = logging.getLogger(__name__)

class Model(ABC):
    def __init__(self, data: np.array) -> None:
        self.data = data
        self.solution = None

    @property
    def aic(self) -> float:
        return 2 * self.loglikelihood + 2 * self.number_of_parameters

    @abstractmethod
    def fit_model(self) -> np.array:
        pass

    @abstractmethod
    def _cost_function(self) -> float:
        pass

    @property
    def initial_solution(self) -> np.array:
        pass

    @abstractmethod
    def run_simulation(self) -> np.array:
        pass

    @abstractmethod
    def transform_to_uniform(self) -> np.array:
        pass

    @abstractmethod
    def transform_to_true(self) -> np.array:
        pass


class MA(Model):
    def __init__(self, data: np.array, q: int):
        super().__init__(data)
        self.q = q

    def fit_model(self):
        inital_solution = self.inital_solution
        solution = minimize(fun=self._cost_function, x0=inital_solution,  method="trust-constr")
        if not solution.success:
            logger.error(f"optimizer success {solution.success}")
        return solution.x

    def _cost_function(self, params: np.array) -> float:
        """
        Defines the conditional log loss for the MA model. Calculates the loss recursively.

        Args:
            params (np.array): The first element is the sigma paramter. The rest is the theta parameters.

        Returns:
            float: log loss value.
        """
        lag = self.q
        number_of_observations = len(self.data)
        residuals = np.zeros(lag)

        mu = params[0]
        sigma = params[1]
        theta = params[2:]

        loss = 0

        for i in range(0, number_of_observations):
            loss += ((self.data[i] - mu - theta.T @ residuals) / sigma ) ** 2
            updated_residual = self.data[i] - ( mu + theta.T @ residuals)
            residuals = np.insert(residuals[:-1], 0, updated_residual)
        loss = loss + number_of_observations * np.log(sigma **  2)

        return loss

    @property
    def inital_solution(self) -> np.array:
        result = np.ones(self.q + 2) * 0.05
        mu = np.mean(self.data)
        sigma = np.std(self.data)
        result[0] = mu
        result[1] = sigma
        return result

    def run_simulation(self) -> np.array:
        pass

    def transform_to_uniform(self) -> np.array:
        pass

    def transform_to_true(self) -> np.array:
        pass

    def _generate_residuals(self):
        raise NotImplemented

    def step_prediction(self):
        pass

class AR(Model):
    pass

class NNAR(Model):
    pass

class ARMA(Model):
    pass

class GARCH(Model):
    pass

class GJRGARCH(Model):
    pass

class ARGARCH(Model):
    pass

class ARGJRGARCH(Model):
    pass

class ARMAGARCH(Model):
    pass
