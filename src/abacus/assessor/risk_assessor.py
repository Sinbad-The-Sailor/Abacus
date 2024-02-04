# -*- coding: utf-8 -*-
import torch
import numpy as np

from scipy.optimize import minimize

from src.abacus.config import EVT_THRESHOLD, GEV_INITIAL_SOLUTION
from src.abacus.utils.portfolio import Portfolio



class RiskAssessor:
    """
    Used for the risk assessment of portfolio returns given an empirical return distribution. Risk assessment is found
    through standard measurements of Value at Risk and Expected Shortfall with user-specified confidence levels. Such
    measurements are calculated with Extreme Value Theory (EVT).

    Portfolio returns are translated into portfolio losses in the class.

    The class is responsible for slicing the return tensor into the shape of the portfolio at the given time step.
    """

    def __init__(self, portfolio: Portfolio, return_tensor: torch.Tensor, time_step: int):
        self._portfolio = portfolio
        self._return_tensor = return_tensor
        self._time_step = time_step
        self._calibrated = False
        self._check_time_step()

    def summary(self):
        headers = "VaR", "Extreme VaR", "ES", "Extreme ES"
        confidence_99 = self.value_at_risk(0.99), self.extreme_value_at_risk(0.99), self.expected_shortfall(0.99), self.extreme_expected_shortfall(0.99)
        confidence_999 = self.value_at_risk(0.999), self.extreme_value_at_risk(0.999), self.expected_shortfall(0.999), self.extreme_expected_shortfall(0.999)
        print(*headers)
        print(*confidence_99)
        print(*confidence_999)

    @property
    def _evt_threshold(self) -> torch.Tensor:
        return torch.quantile(self._portfolio_loss_scenarios, torch.tensor(EVT_THRESHOLD))

    @property
    def _excess_portfolio_losses(self) -> torch.Tensor:
        large_losses = self._portfolio_loss_scenarios[self._portfolio_loss_scenarios > self._evt_threshold]
        excess_losses = large_losses - self._evt_threshold
        return excess_losses

    @property
    def _return_matrix(self) -> torch.Tensor:
        return self._return_tensor[:,self._time_step,:]

    @property
    def _reduced_return_matrix(self) -> torch.Tensor:
        sorted_ids = torch.tensor(self._portfolio.indices)
        return torch.index_select(self._return_matrix, dim=0, index=sorted_ids)

    @property
    def _maxmimum_time_step(self):
        return self._return_tensor.size(dim=1)

    @property
    def _weights(self) -> torch.Tensor:
        portfolio_weights = self._portfolio.weights
        sorted_keys = sorted(list(portfolio_weights.keys()), key=lambda x: x.id)
        weights = torch.empty(len(portfolio_weights))
        for i, key in enumerate(sorted_keys):
            weights[i] = portfolio_weights[key]
        return weights

    @property
    def _portfolio_return_scenarios(self) -> torch.Tensor:
        return self._reduced_return_matrix.T @ self._weights

    @property
    def _portfolio_loss_scenarios(self) -> torch.Tensor:
        return -self._portfolio_return_scenarios

    @property
    def _constraints(self) -> list[dict]:
        constraints = []
        observations = self._excess_portfolio_losses
        constraints.append({"type": "ineq", "fun": lambda x: x[1]})

        for observation in observations:
            constraints.append({"type": "ineq", "fun": lambda x: 1 + x[0] / x[1] * observation})

        return constraints

    def _check_time_step(self):
        if self._time_step > self._maxmimum_time_step:
            raise ValueError(f"Time step {self._time_step} is above maximum: {self._maxmimum_time_step}.")

    def _calibrate(self):
        solution = minimize(self._cost_function,
                            np.array(GEV_INITIAL_SOLUTION),
                            jac=True,
                            constraints=self._constraints)
        self._solution = solution
        self._parameters = solution.x
        self._calibrated = True

    def _cost_function(self, parameters: np.array) -> tuple[float, 2]:
        """
        Internal log-loss function. Note, this returns the log-loss with associated gradient.

        Args:
            params: parameters for the generalized pareto distribution
            data: observations above threshold determined in EVT.

        Returns: the log-loss value and the gradient of the generalized pareto distribution.
        """
        parameters = torch.tensor(parameters, requires_grad=True)
        log_loss = 0
        observations = self._excess_portfolio_losses
        number_of_observations = len(observations)

        for observation in observations:
            log_loss = log_loss + torch.log(1 + parameters[0] / parameters[1] * observation)

        log_loss = number_of_observations * torch.log(parameters[1]) + (1 + 1 / parameters[0]) * log_loss

        log_loss.backward()
        return log_loss.data.cpu().numpy(), parameters.grad.data.cpu().numpy()


    def extreme_value_at_risk(self, confidence_level: float) -> float:
        """
        Calculates the Value at Risk of portfolio returns (either in percentage terms or in absolute value) as specified
        in the instantiation of the RiskAssessor, based on EVT.

        Args:
            quantile: the quantile of the Value at Risk measurement. Note, portfolio losses are specified!

        Returns: EVT based Value at Risk for a given confidence level.
        """
        if not self._calibrated:
            self._calibrate()
        confidence_level = torch.tensor(confidence_level)
        total_number_of_observations = self._portfolio_loss_scenarios.size(dim=0)
        number_of_excess_observations = self._excess_portfolio_losses.size(dim=0)
        threshold = self._evt_threshold
        parameters = torch.tensor(self._parameters)
        xi, beta = parameters[0], parameters[1]

        extreme_var = threshold + beta / xi * ((total_number_of_observations / number_of_excess_observations * (1 - confidence_level)) ** (-xi) - 1)

        return extreme_var.item()


    def extreme_expected_shortfall(self, confidence_level: float) -> float:
        """
        Calculates the Expected Shortfall of portfolio returns (either in percentage terms or in absolute value) as
        specified in the instantiation of the RiskAssessor, based on EVT.

        Args:
            quantile: the quantile of the Value at Risk measurement. Note, portfolio losses are specified!

        Returns: EVT based Expected Shortfall for a given confidence level.
        """
        if not self._calibrated:
            self._calibrate()
        extreme_var = torch.tensor(self.extreme_value_at_risk(confidence_level))
        threshold = self._evt_threshold
        parameters = torch.tensor(self._parameters)
        xi, beta = parameters[0], parameters[1]
        extreme_es = extreme_var / (1 - xi) + (beta - threshold * xi) /(1 - xi)

        return extreme_es.item()


    def value_at_risk(self, confidence_level: float) -> float:
        """
        Calculates the Value at Risk of portfolio returns (either in percentage terms or in absolute value) as specified
        in the instantiation of the RiskAssessor.

        Args:
            quantile: the quantile of the Value at Risk measurement. Note, portfolio losses are specified!

        Returns: Value at Risk for a given confidence level.
        """
        confidence_level = torch.tensor(confidence_level)
        return torch.quantile(self._portfolio_loss_scenarios, confidence_level).item()

    def expected_shortfall(self, confidence_level: float) -> float:
        """
        Calculates the Expected Shortfall of portfolio returns (either in percentage terms or in absolute value) as
        specified in the instantiation of the RiskAssessor.

        Args:
            quantile: the quantile of the Value at Risk measurement. Note, portfolio losses are specified!

        Returns: Expected Shortfall for a given confidence level.
        """
        var = self.value_at_risk(confidence_level)
        losses_greater_than_var = self._portfolio_loss_scenarios[self._portfolio_loss_scenarios > var]
        return torch.mean(losses_greater_than_var).item()
