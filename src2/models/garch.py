# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

from scipy.optimize import minimize

from models.model import Model
from utils.config import INITIAL_VARIANCE_GARCH_OBSERVATIONS, INITIAL_GARCH_PARAMETERS

class GARCH(Model):

    def __init__(self, time_series: pd.Series):
        super().__init__(time_series)
        self._data = np.array(self._data)
    @property
    def mse(self):
        ...

    def calibrate(self):
        self._initiate_parameters()
        self._inital_solution = self._precondition_parameters(parameters=INITIAL_GARCH_PARAMETERS)
        solution = minimize(self._cost_function,
                            self._inital_solution,
                            method="L-BFGS-B")
        self._solution = solution
        self._calibrated = True

    def transform_to_true(self, uniform_sample: torch.Tensor) -> torch.Tensor:
        ...

    def transform_to_uniform(self):
        ...

    def _compute_variance(self, parameters: torch.Tensor) -> torch.Tensor:
        initial_variance = self._initial_variance
        variance = np.zeros(self._number_of_observations)
        mu_corr = np.exp(-np.exp(-parameters[0]))
        mu_ewma = np.exp(-np.exp(-parameters[1]))

        for i in range(self._number_of_observations):
            if i == 0:
                variance[i] = initial_variance
            else:
                variance[i] = self._long_run_variance + mu_corr * (mu_ewma * variance[i - 1]
                                                                   + (1 - mu_ewma) * self._squared_returns[i - 1]
                                                                   - self._long_run_variance
                                                                   )

        from matplotlib import pyplot as plt
        plt.plot(np.sqrt(variance))
        plt.show()
        return variance


    def _cost_function(self, parameters: torch.Tensor) -> float:
        """
        Defines the conditional log loss for the GARCH model. Calculates the loss recursively.

        Args:
            parameters (torch.Tensor): parameters formatted as [alpha, beta].

        Returns:
            float: log loss value.
        """

        variance = self._compute_variance(parameters=parameters)
        log_loss = np.sum(np.log(variance) + self._squared_returns / variance)
        print(log_loss)
        return log_loss

    def _gradient(self, parameters: torch.Tensor) -> torch.Tensor:

        mu_corr = np.exp(-np.exp(-parameters[0]))
        mu_ewma = np.exp(-np.exp(-parameters[1]))
        variance = self._compute_variance(parameters=parameters)
        squared_returns = self._squared_returns

        partial_mu_corr = self._partial_mu_corr(mu_corr, mu_ewma, variance, squared_returns)
        partial_z_corr = mu_corr * np.exp(-parameters[0]) * np.sum((1 / variance - squared_returns / variance ** 2 * partial_mu_corr))

        partial_mu_ewma = self._partial_mu_ewma(mu_ewma, variance, squared_returns)
        partial_z_ewma = mu_corr * mu_ewma * np.exp(-parameters[1]) * np.sum((1 / variance - squared_returns / variance ** 2 * partial_mu_ewma))
        print("gradient:", [partial_z_corr, partial_z_ewma])
        return partial_z_corr, partial_z_ewma

    def _partial_mu_corr(self, mu_corr: torch.Tensor, mu_ewma: torch.Tensor, variance: torch.Tensor, squared_returns: torch.Tensor) -> torch.Tensor:
        inital_partial = 0
        long_run_variance = self._long_run_variance
        partials = np.zeros(self._number_of_observations)

        for i in range(self._number_of_observations):
            if i == 0:
                partials[i] = inital_partial
            else:
                partials[i] = mu_ewma * variance[i-1] + (1 - mu_ewma) * squared_returns[i-1] - long_run_variance + mu_corr * mu_ewma * partials[i - 1]

        return partials

    def _partial_mu_ewma(self, mu_ewma: torch.Tensor, variance: torch.Tensor, squared_returns: torch.Tensor) -> torch.Tensor:
        initial_partial = 0
        partials = np.zeros(self._number_of_observations)

        for i in range(self._number_of_observations):
            if i == 0:
                partials[i] = initial_partial
            else:
                partials[i] = variance[i-1] + mu_ewma * partials[i-1] - squared_returns[i-1]

        return partials

    def _initiate_parameters(self):
        self._squared_returns = self._data ** 2
        self._initial_squared_returns = self._squared_returns[0]
        self._initial_variance = self._compute_inital_variance()
        self._long_run_variance = np.std(self._data) ** 2

        print("inital vol", np.sqrt(self._initial_variance))
        print("long vol:", np.sqrt(self._long_run_variance))


    def _compute_inital_variance(self):
        if self._number_of_observations > INITIAL_VARIANCE_GARCH_OBSERVATIONS:
            return np.std(self._data[:INITIAL_VARIANCE_GARCH_OBSERVATIONS]) ** 2
        return self._initial_squared_returns

    @staticmethod
    def _precondition_parameters(parameters: np.array) -> np.array:
        """
        Preconditioning to obtain more stable optimzation problem.

        Args:
            parameters (torch.Tensor): GARCH parameters.

        Returns:
            torch.Tesnor: transformed GARCH parameters.
        """
        mu_corr = parameters[0] + parameters[1]
        mu_ewma = parameters[1] / (parameters[0] + parameters[1])

        z_corr = np.log(-1 / np.log(mu_corr))
        z_ewma = np.log(-1 / np.log(mu_ewma))

        return np.array([z_corr, z_ewma])

    @staticmethod
    def _uncondition_parameters(params: torch.Tensor) -> torch.Tensor:
        """
        Unconditioning to obtain more original parameters from transformed parameters.

        Args:
            params (torch.Tensor): transformed GARCH parameters.

        Returns:
            torch.Tensor: GARCH parameters.
        """
        mu_corr = np.exp(-np.exp(-params[0]))
        mu_ewma = np.exp(-np.exp(-params[1]))

        alpha = mu_corr * (1 - mu_ewma)
        beta = mu_corr * mu_ewma

        return np.array([alpha, beta])
