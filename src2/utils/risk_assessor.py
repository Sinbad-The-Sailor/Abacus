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
        return torch.index_select(self._return_matrix, dim=0, index=sorted_ids)

    @property
    def _maxmimum_time_step(self):
        return self._return_tensor.size(dim=1)

    @property
    def _weights(self) -> torch.Tensor:
        portfolio_weights = self._portfolio.weights
        weights = torch.empty(len(portfolio_weights))
        sorted_keys = sorted(list(portfolio_weights), key=lambda x: x.id)
        for i, key in enumerate(sorted_keys):
            weights[i] = portfolio_weights[key]
        return weights

    @property
    def _portfolio_return_scenarios(self) -> torch.Tensor:
        return self._reduced_return_matrix.T @ self._weights

    @property
    def _portfolio_loss_scenarios(self) -> torch.Tensor:
        return -self._portfolio_return_scenarios

    def _check_time_step(self):
        if self._time_step > self._maxmimum_time_step:
            raise ValueError(f"Time step {self._time_step} is above maximum: {self._maxmimum_time_step}.")

    def extreme_value_at_risk(self, confidence_level: float) -> float:
        ...

    def extreme_expected_shortfall(self, confidence_level: float) -> float:
        ...

    def value_at_risk(self, confidence_level: float) -> float:
        confidence_level = torch.tensor(confidence_level)
        return torch.quantile(self._portfolio_loss_scenarios, confidence_level)

    def expected_shortfall(self, confidence_level: float) -> float:
        confidence_level = torch.tensor(confidence_level)
        var = self.value_at_risk(confidence_level)
        losses_greater_than_var = torch.le(self._portfolio_loss_scenarios, var)
        return torch.mean(losses_greater_than_var)
