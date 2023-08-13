# -*- coding: utf-8 -*-
import torch

from portfolio import Portfolio



class RiskAssessor:

    def __init__(self, portfolio: Portfolio, return_tensor: torch.Tensor, time_step: int):
        self._portfolio = portfolio
        self._return_tensor = return_tensor
        self._time_step = time_step

    @property
    def _return_matrix(self) -> torch.Tensor:
        return self._return_tensor[:,self._time_step,:]

    @property
    def _reduced_return_matrix(self) -> torch.Tensor:
        return None

    def _weights(self) -> torch.Tensor:
        portfolio_weights = self._portfolio.weights
        number_of_assets = len(portfolio_weights)
        weights = torch.empty(number_of_assets)

        # Weights sorted and by id in array.

        return weights

    def _portfolio_scenarios(self) -> torch.Tensor:
        return self._reduced_return_matrix.T @ self._weights

    def extreme_value_at_risk(self, confidence_level: float) -> float:
        ...

    def extreme_expected_shortfall(self, confidence_level: float) -> float:
        ...

    def normal_value_at_risk(self, confidence_level: float) -> float:
        ...

    def normal_expected_shortfall(self, confidence_level: float) -> float:
        ...

    def summary(self):
        ...
