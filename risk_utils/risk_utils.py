import numpy as np


class RiskAssessor:
    def __init__(self, portfolio_returns: list):
        self._portfolio_losses = -portfolio_returns

    def value_at_risk_non_parametric(self, quantile: float) -> float:
        return np.quantile(self._portfolio_losses, quantile)

    def expected_shortfall_non_parametric(self):
        pass

    def value_at_risk_evt(self):
        pass

    def expected_shortfall_evt(self):
        pass

    def evt_parameter_generator(self) -> list:
        pass
