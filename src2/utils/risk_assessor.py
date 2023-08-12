# -*- coding: utf-8 -*-
import torch

from portfolio import Portfolio


class RiskAssessor:

    def __init__(self, return_tensor: torch.Tensor, time_period: int , scaling: int):
        ...

    def _weights(self) -> torch.Tensor:
        ...

    def extreme_value_at_risk(self, confidence_level: float):
        ...

    def extreme_expected_shortfall(self, confidence_level: float):
        ...

    def value_at_risk(self, confidence_level: float):
        ...

    def expected_shortfall(self, confidence_level: float):
        ...
