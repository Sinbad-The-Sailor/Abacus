# -*- coding: utf-8 -*-
import torch

from utils.portfolio import Portfolio



class RiskAssessor:

    def __init__(self, portfolio: Portfolio, return_tensor: torch.Tensor, time_step: int):
        self._portfolio = portfolio
        self._return_tensor = return_tensor
        self._time_step = time_step

        self._check_time_step()

    @property
    def _return_matrix(self) -> torch.Tensor:
        return self._return_tensor[:,self._time_step,:]

    @property
    def _reduced_return_matrix(self) -> torch.Tensor:
        instrument_ids = [instrument.id for instrument in self._portfolio.weights]
        sorted_ids = torch.tensor(sorted(instrument_ids))
        return torch.take(self._return_matrix, sorted_ids)

    @property
    def _maxmimum_time_step(self):
        return self._return_tensor.size(dim=1)

    def _weights(self) -> torch.Tensor:
        portfolio_weights = self._portfolio.weights
        weights = torch.empty(len(portfolio_weights))
        sorted_keys = sorted(list(portfolio_weights), key=lambda x: x.id)
        for i, key in enumerate(sorted_keys):
            weights[i] = portfolio_weights[key]
        return weights

    def _portfolio_scenarios(self) -> torch.Tensor:
        return self._reduced_return_matrix.T @ self._weights

    def _check_time_step(self):
        if self._time_step > self._maxmimum_time_step:
            raise ValueError(f"Time step {self._time_step} is above maximum: {self._maxmimum_time_step}.")

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
