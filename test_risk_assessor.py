import numpy as np

from scipy.optimize import minimize


class RiskAssessor:
    """
    Used for the risk assessment of portfolio returns given an empirical return distribution. Risk assessment is found
    through standard measurements of Value at Risk and Expected Shortfall with user-specified confidence levels. Such
    measurements are calculated with Extreme Value Theory (EVT).

    Portfolio returns are translated into portfolio losses in the class.
    """

    def __init__(self, portfolio_returns: list):
        self._portfolio_losses = -portfolio_returns

        self._evt_threshold = np.quantile(self._portfolio_losses, 0.95)
        self._excess_losses = [loss - self._evt_threshold for loss in self._portfolio_losses
                               if loss > self._evt_threshold]
        self._evt_params = np.zeros(2)

    def value_at_risk_non_parametric(self, quantile: float) -> float:
        """
        Calculates the Value at Risk of portfolio returns (either in percentage terms or in absolute value) as specified
        in the instantiation of the RiskAssessor.

        Args:
            quantile: the quantile of the Value at Risk measurement. Note, portfolio losses are specified!

        Returns: Value at Risk for a given confidence level.
        """
        return np.quantile(self._portfolio_losses, quantile)

    def expected_shortfall_non_parametric(self, quantile: float) -> float:
        """
        Calculates the Expected Shortfall of portfolio returns (either in percentage terms or in absolute value) as
        specified in the instantiation of the RiskAssessor.

        Args:
            quantile: the quantile of the Value at Risk measurement. Note, portfolio losses are specified!

        Returns: Expected Shortfall for a given confidence level.
        """
        var = np.quantile(self._portfolio_losses, quantile)
        return np.mean([x for x in self._portfolio_losses if x >= var])

    def value_at_risk_evt(self, quantile: float) -> float:
        """
        Calculates the Value at Risk of portfolio returns (either in percentage terms or in absolute value) as specified
        in the instantiation of the RiskAssessor, based on EVT.

        Args:
            quantile: the quantile of the Value at Risk measurement. Note, portfolio losses are specified!

        Returns: EVT based Value at Risk for a given confidence level.
        """
        if not self._evt_params.any():
            self._evt_params = self._evt_parameter_generator()

        total_n_observations = len(self._portfolio_losses)
        excess_n_observations = len(self._excess_losses)
        threshold = self._evt_threshold
        xi, beta = self._evt_params[0], self._evt_params[1]

        return threshold + beta / xi * ((total_n_observations / excess_n_observations * (1 - quantile)) ** (-xi) - 1)

    def expected_shortfall_evt(self, quantile: float) -> float:
        """
        Calculates the Expected Shortfall of portfolio returns (either in percentage terms or in absolute value) as
        specified in the instantiation of the RiskAssessor, based on EVT.

        Args:
            quantile: the quantile of the Value at Risk measurement. Note, portfolio losses are specified!

        Returns: EVT based Expected Shortfall for a given confidence level.
        """
        if not self._evt_params.any():
            self._evt_params = self._evt_parameter_generator()

        var = self.value_at_risk_evt(quantile)
        threshold = self._evt_threshold
        xi, beta = self._evt_params[0], self._evt_params[1]

        return var / (1 - xi) + (beta - threshold * xi) / (1 - xi)

    def risk_summary(self):
        """
        Prints a summary of risk measurements.
        """
        var_np_95 = self.value_at_risk_non_parametric(0.95)
        var_np_99 = self.value_at_risk_non_parametric(0.99)
        es_np_95 = self.expected_shortfall_non_parametric(0.95)
        es_np_99 = self.expected_shortfall_non_parametric(0.99)
        var_evt_99 = self.value_at_risk_evt(0.99)
        es_evt_99 = self.expected_shortfall_evt(0.99)

        print('======== RISK ASSESSMENT ========')
        print(f'VaR95 {var_np_95},    ES95 {es_np_95}')
        print(f'VaR99 {var_np_99},     ES99 {es_np_99}')
        print(f'EVT VaR99 {var_evt_99}, EVT ES99 {es_evt_99}')

    def _evt_parameter_generator(self) -> list:
        """
        Estimates generalized pareto parameters used for Value at Risk and Expected Shortfall calculations based on EVT.

        Returns: list of parameters, xi and beta, for a generalized pareto distribution.
        """
        cons = []
        for obs in self._excess_losses:
            cons.append(
                {'type': 'ineq', 'fun': lambda x: 1 + x[0] / x[1] * obs})

        x0 = [0.15, 0.01]
        sol = minimize(self._evt_ml_objective_function, x0,
                       constraints=cons, args=self._excess_losses)
        return sol.x

    @staticmethod
    def _evt_ml_objective_function(params, data) -> float:
        """
        Internal log-likelihood function. Note, this returns the negative of the log-likelihood.

        Args:
            params: parameters for the generalized pareto distribution
            data: observations above threshold determined in EVT.

        Returns: the negative log-likelihood value of generalized pareto distribution.
        """

        n_observations = len(data)
        log_likelihood = 0

        for obs in data:
            log_likelihood = log_likelihood + \
                np.log(1 + params[0] / params[1] * obs)

        return n_observations * np.log(params[1]) + (1 + 1 / params[0]) * log_likelihood
