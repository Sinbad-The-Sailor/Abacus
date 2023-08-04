# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from torch.distributions import Normal

from models.model import Model
from utils.config import INITIAL_VARIANCE_GARCH_OBSERVATIONS, INITIAL_GARCH_PARAMETERS
from utils.exceptions import ParameterError

class GARCH(Model):

    def __init__(self, time_series: pd.Series):
        super().__init__(time_series)

    @property
    def parameters(self) -> tuple[float, 3]:
        self._check_calibration()

        optimal_parameters = self._optimal_parameters
        alpha = optimal_parameters[0]
        beta = optimal_parameters[1]
        long_run_variance = self._long_run_variance

        return alpha, beta, long_run_variance

    @property
    def _number_of_parameters(self) -> int:
        return len(self.parameters)

    @property
    def _optimal_parameters(self) -> np.array:
        return self._uncondition_parameters(parameters=torch.tensor(self._solution.x)).numpy()

    @property
    def _inital_solution(self) -> np.array:
        return np.array(self._precondition_parameters(parameters=torch.tensor(INITIAL_GARCH_PARAMETERS)))

    @property
    def _log_likelihood(self) -> torch.tensor:
        optimal_cost = torch.tensor(self._solution.fun)
        pi = torch.tensor(torch.pi)
        number_of_observations = torch.tensor(self._number_of_observations)
        return -(1 / 2) * (number_of_observations * torch.log(2 * pi) + optimal_cost)

    def calibrate(self):
        self._initiate_parameters()
        self._solve_maximum_likelihood()
        self._sanity_check()
        self._calibrated = True

    def transform_to_true(self, uniform_sample: torch.Tensor) -> torch.Tensor:
        self._check_calibration()

        number_of_samples = len(uniform_sample)
        normals = Normal(0,1).icdf(uniform_sample)
        simulated_values = torch.zeros(number_of_samples)
        parameters = torch.tensor(self._solution.x)
        mu_corr, mu_ewma = self._intermediary_parameters(parameters=parameters)


        variance = self._compute_variance(parameters=torch.tensor(self._solution.x))[-1]
        squared_return = self._squared_returns[-1]

        for i in range(number_of_samples):
            variance = self._update_variance(variance, squared_return, mu_corr, mu_ewma)
            return_ = torch.sqrt(variance) * normals[i]
            squared_return = torch.square(return_)
            simulated_values[i] = return_

        return simulated_values


    def transform_to_uniform(self):
        self._check_calibration()

        parameters = torch.Tensor(self._solution.x)
        variance = self._compute_variance(parameters=parameters)
        volatility = torch.sqrt(variance)
        returns = self._data
        residuals = returns / volatility

        return Normal(0, 1).cdf(residuals)

    def _cost_function(self, parameters: np.array) -> tuple[float, 2]:
        """
        Defines the conditional log loss for the GARCH model. Calculates the loss recursively.

        Args:
            parameters (np.array): parameters formatted as [z_corr, z_ewma].

        Returns:
            tuple(float, float): log loss value and the corresponding gradient.
        """
        parameters = torch.tensor(parameters, requires_grad=True)
        mu_corr, mu_ewma = self._intermediary_parameters(parameters=parameters)

        log_loss = torch.tensor(0.0)
        for i in range(self._number_of_observations):
            if i == 0:
                variance = self._initial_variance
            else:
                variance = self._update_variance(variance, self._squared_returns[i - 1].detach(), mu_corr, mu_ewma)

            log_loss = log_loss + torch.log(variance) + self._squared_returns[i].detach() / variance

        log_loss.backward()
        return log_loss.data.cpu().numpy(), parameters.grad.data.cpu().numpy()

    def _initiate_parameters(self):
        self._squared_returns = self._data ** 2
        self._initial_squared_returns = self._squared_returns[0]
        self._initial_variance = self._compute_inital_variance()
        self._long_run_variance = torch.square(torch.std(self._data))

    def _solve_maximum_likelihood(self):
        solution = minimize(self._cost_function,
                                    self._inital_solution,
                                    method="L-BFGS-B",
                                    jac=True)
        self._solution = solution

    def _compute_inital_variance(self) -> torch.Tensor:
        if self._number_of_observations > INITIAL_VARIANCE_GARCH_OBSERVATIONS:
            return torch.square(torch.std(self._data[:INITIAL_VARIANCE_GARCH_OBSERVATIONS]))
        return self._initial_squared_returns

    def _compute_variance(self, parameters: torch.Tensor) -> torch.Tensor:
        initial_variance = self._initial_variance
        variances = torch.zeros(self._number_of_observations)
        mu_corr, mu_ewma = self._intermediary_parameters(parameters=parameters)

        for i in range(self._number_of_observations):
            if i == 0:
                variance = initial_variance
            else:
                variance = self._update_variance(variance, self._squared_returns[i-1], mu_corr, mu_ewma)
            variances[i] = variance
        return variances

    def _update_variance(self, variance: torch.Tensor, squared_return: torch.Tensor, mu_corr, mu_ewma):
        return self._long_run_variance + mu_corr * (mu_ewma * variance + (1 - mu_ewma) * squared_return - self._long_run_variance)

    def _sanity_check(self):
        parameter_check = self._check_parameters()
        solution_check = self._check_solution()

        if not parameter_check:
            # log.
            ...

        if not solution_check:
            # log.
            ...

        if not parameter_check or not solution_check:
            raise ParameterError("Parameters could not be asceratined succesfully.")

    def _check_parameters(self) -> bool:
        return self._solution.success

    def _check_solution(self) -> bool:
        return np.sum(self._optimal_parameters) < 1

    @staticmethod
    def _intermediary_parameters(parameters: torch.Tensor):
        """Computes mu_corr and mu_ewma from z_corr and z_ewma.

        Args:
            parameters (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        mu = torch.exp(-torch.exp(-parameters))
        return mu[0], mu[1]

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
    def _uncondition_parameters(parameters: torch.Tensor) -> torch.Tensor:
        """
        Unconditioning to obtain more original parameters from transformed parameters.

        Args:
            params (torch.Tensor): transformed GARCH parameters.

        Returns:
            torch.Tensor: GARCH parameters.
        """
        mu_corr = torch.exp(-torch.exp(-parameters[0]))
        mu_ewma = torch.exp(-torch.exp(-parameters[1]))

        alpha = mu_corr * (1 - mu_ewma)
        beta = mu_corr * mu_ewma

        return torch.tensor([alpha, beta])
