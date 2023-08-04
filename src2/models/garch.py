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

    def calibrate(self):
        self._initiate_parameters()
        self._inital_solution = np.array(self._precondition_parameters(parameters=torch.tensor(INITIAL_GARCH_PARAMETERS)))
        solution = minimize(self._cost_function,
                            self._inital_solution,
                            method="BFGS",
                            jac=True)
        self._solution = solution
        self._calibrated = True

    def transform_to_true(self, uniform_sample: torch.Tensor) -> torch.Tensor:
        ...

    def transform_to_uniform(self):
        ...

    def _compute_variance(self, parameters: torch.Tensor) -> torch.Tensor:
        initial_variance = self._initial_variance
        variance = np.zeros(self._number_of_observations)
        mu_corr = torch.exp(-torch.exp(-parameters[0]))
        mu_ewma = torch.exp(-torch.exp(-parameters[1]))

        for i in range(self._number_of_observations):
            if i == 0:
                variance[i] = initial_variance
            else:
                variance[i] = self._long_run_variance + mu_corr * (mu_ewma * variance[i - 1]
                                                                   + (1 - mu_ewma) * self._squared_returns[i - 1]
                                                                   - self._long_run_variance
                                                                   )

        return torch.tensor(variance)

    def _cost_function(self, parameters: np.array) -> tuple[float, 2]:
        """
        Defines the conditional log loss for the GARCH model. Calculates the loss recursively.

        Args:
            parameters (np.array): parameters formatted as [z_corr, z_ewma].

        Returns:
            tuple(float, float): log loss value and the corresponding gradient.
        """
        parameters = torch.tensor(parameters, requires_grad=True)
        variance = self._compute_variance(parameters=parameters)
        log_loss = torch.sum(torch.log(variance) + self._squared_returns / variance)
        log_loss.backward()
        print(log_loss, parameters.grad)

        return log_loss.data.cpu().numpy(), parameters.grad.data.cpu().numpy()

    def _initiate_parameters(self):
        self._squared_returns = self._data ** 2
        self._initial_squared_returns = self._squared_returns[0]
        self._initial_variance = self._compute_inital_variance()
        self._long_run_variance = torch.square(torch.std(self._data))

    def _compute_inital_variance(self) -> torch.Tensor:
        if self._number_of_observations > INITIAL_VARIANCE_GARCH_OBSERVATIONS:
            return torch.square(torch.std(self._data[:INITIAL_VARIANCE_GARCH_OBSERVATIONS]))
        return self._initial_squared_returns

    @property
    def _number_of_parameters(self):
        return super()._number_of_parameters

    @property
    def _log_likelihood(self):
        return super()._log_likelihood

    @staticmethod
    def _precondition_parameters(parameters: torch.Tensor) -> torch.Tensor:
        """
        Preconditioning to obtain more stable optimzation problem.

        Args:
            parameters (torch.Tensor): GARCH parameters.

        Returns:
            torch.Tensor: transformed GARCH parameters.
        """
        mu_corr = parameters[0] + parameters[1]
        mu_ewma = parameters[1] / (parameters[0] + parameters[1])

        z_corr = torch.log(-1 / torch.log(mu_corr))
        z_ewma = torch.log(-1 / torch.log(mu_ewma))

        return torch.tensor([z_corr, z_ewma])

    @staticmethod
    def _uncondition_parameters(params: torch.Tensor) -> torch.Tensor:
        """
        Unconditioning to obtain more original parameters from transformed parameters.

        Args:
            params (torch.Tensor): transformed GARCH parameters.

        Returns:
            torch.Tensor: GARCH parameters.
        """
        mu_corr = torch.exp(-torch.exp(-params[0]))
        mu_ewma = torch.exp(-torch.exp(-params[1]))

        alpha = mu_corr * (1 - mu_ewma)
        beta = mu_corr * mu_ewma

        return torch.tensor([alpha, beta])
