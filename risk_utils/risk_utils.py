import numpy as np


class RiskAssessor:
    """
    Used for the risk assessment of portfolio returns given an empirical return distribution. Risk assessment is found
    through standard measurements of Value at Risk and Expected Shortfall with user-specified confidence levels. Such
    measurements are calculated with Extreme Value Theory (EVT).

    Portfolio returns are translated into portfolio losses in the class.
    """

    def __init__(self, portfolio_returns: list):
        self._portfolio_losses = -portfolio_returns

    def value_at_risk_non_parametric(self, quantile: float) -> float:
        """
        Calculates the Value at Risk of portfolio returns (either in percentage terms or in absolute value) as specified
        in the instantiation of the RiskAssessor.

        Args:
            quantile: The quantile of the Value at Risk measurement. Note, portfolio losses are specified!

        Returns: Value at Risk for a given confidence level.
        """
        return np.quantile(self._portfolio_losses, quantile)

    def expected_shortfall_non_parametric(self, quantile: float) -> float:
        """
        Calculates the Expected Shortfall of portfolio returns (either in percentage terms or in absolute value) as
        specified in the instantiation of the RiskAssessor.

        Args:
            quantile: The quantile of the Value at Risk measurement. Note, portfolio losses are specified!

        Returns: Expected Shortfall for a given confidence level.
        """
        pass

    def value_at_risk_evt(self, quantile: float) -> float:
        pass

    def expected_shortfall_evt(self, quantile: float) -> float:
        pass

    def _evt_parameter_generator(self) -> list:
        pass

    def _evt_ml_objective_function(self) -> float:
        pass
