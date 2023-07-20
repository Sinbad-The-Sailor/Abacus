# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from torch.distributions import Normal

from price_history import PriceHistory
from config import MAXIMUM_AR_ORDER


class StationarityError(ValueError):
    ...

class ParameterError(ValueError):
    ...


class Model(ABC):
    time_series: pd.Series
    _calibrated: bool
    _data: torch.Tensor


    @abstractmethod
    def calibrate(self):
        ...

    @abstractmethod
    def mse(self):
        ...

    @abstractmethod
    def transform_to_uniform(self):
        ...

    @abstractmethod
    def transform_to_true(self, uniform_sample: torch.Tensor) -> torch.Tensor:
        ...



class AR(Model):
    """The Auto Regressive AR(p) model.

    Parameter estimation with statistical estimators for mean and standard devations.

    Least Squares Solution of prediction for the /phi parameters w. QR decomposition.

    Lazy MSE calculation.

    Lag is determined by PACF and the

    Unit roots are checked.

    Stationarity is not checked.

    """

    def __init__(self, time_series: pd.Series):
        self.time_series = time_series
        self._data = torch.tensor(time_series.values)
        self._calibrated = False
        self._number_of_observations = len(time_series)

    @property
    def mse(self):
        self._check_calibration()

        predictions = self._predictions
        data = self._data[:-self._order]
        difference = predictions - data

        return (1 / self._number_of_observations) * torch.dot(difference, difference)

    @property
    def parameters(self):
        self._check_calibration()

        return [self._phi, self._mu, self._sigma]

    @property
    def _predictions(self):
        return self._Q @ (self._R @ self._phi)


    def calibrate(self):
        self._build_lag_order()
        self._build_data_matricies()
        self._solve_least_squares()
        self._estimate_mean_and_standard_deviation()
        self._estimate_drift()
        self._check_unit_roots()

        self._calibrated = True


    def transform_to_true(self, uniform_sample: torch.Tensor) -> torch.Tensor:
        self._check_calibration()

        order = self._order
        number_of_samples = len(uniform_sample)

        normals = Normal(0,1).icdf(uniform_sample)

        simulated_values = torch.zeros(number_of_samples + order)
        simulated_values[:order] = self._data[:order]


        phi = self._phi
        drift = self._drift
        sigma = self._sigma




        for i in range(number_of_samples):
            past_values = simulated_values[:]


        return simulated_values[self._order:]


    def transform_to_uniform(self) -> torch.Tensor:
        self._check_calibration()
        residuals = self._build_residuals()

        return Normal(0, 1).cdf(residuals)


    def _check_calibration(self):
        """
        Checks if successful calibration has been made.

        Raises:
            ParameterError: if succesful calibration was not made.
        """
        if not self._calibrated:
            raise ParameterError("Model has not been calibrated succesfully.")


    def _check_unit_roots(self):
        """
        Checks for unit roots outside the unit circle.

        Raises:
            StationarityError: raised if unit root is found.
        """
        coefficients = torch.ones(self._order + 1)
        coefficients[1:] = -self._phi
        roots = np.roots(coefficients)

        for root in roots:
            if np.abs(root) >= 1:
                raise StationarityError(f"Root {root} outside of complex unit circle.")


    def _estimate_mean_and_standard_deviation(self):
        self._mu = torch.mean(self._data)
        self._sigma = torch.std(self._data)


    def _estimate_drift(self):
        self._drift = self._mu * (1 - torch.sum(self._phi))


    def _solve_least_squares(self):
        self._phi = torch.linalg.solve(self._R, self._Q.T @ self._b)


    def _build_data_matricies(self):
        self._b = self._data[:-self._order]

        cols = []
        for i in range(1, self._order + 1):

            if i - self._order == 0:
                col = self._data[i:]
            else:
                col = self._data[i: i - self._order]

            cols.append(col)

        A = torch.column_stack(cols)
        self._Q, self._R = torch.linalg.qr(A)

    def _build_lag_order(self):
        """
        The critical value is based on [source] and represents a 95 % confidence interval.
        """

        critical_value = 2 / np.sqrt(self._number_of_observations)

        for p in range(1, MAXIMUM_AR_ORDER + 1):
            self._order = p
            self._build_data_matricies()
            self._solve_least_squares()

            partial_auto_correlation = self._phi[p - 1]

            if torch.abs(partial_auto_correlation) > critical_value:
                self._order = p
            else:
                break

    def _build_residuals(self) -> torch.Tensor:
        predictions = self._predictions
        data = self._data[:-self._order]

        predictable_residuals = (data - predictions) / self._sigma

        non_predicatble_data = self._data[self._number_of_observations - self._order:]
        non_predicable_residuals = (non_predicatble_data - self._mu) / self._sigma

        residuals = torch.cat([predictable_residuals, non_predicable_residuals])
        return residuals






class RiskFactor:
    def __init__(self, identifier: str, price_history: PriceHistory):
        self.identifier = identifier
        self.price_history = price_history

    def __str__(self) -> str:
        return f"{self.identifier}"

    def __repr__(self) -> str:
        return f"{self.identifier}"


class Asset(ABC):
    identifier: str
    risk_factors: list[RiskFactor]

    @property
    @abstractmethod
    def price(self):
        ...

    def __str__(self) -> str:
        return f"Asset: {self.identifier} ({type(self).__name__}) \nRisk Factors: {self.risk_factors}"


class Stock(Asset):
    def __init__(self, identifier, risk_factors):
        self.identifier = identifier
        self.risk_factors = risk_factors

    @property
    def price(self):
        ...
