# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

from torch.distributions import Normal

from src.abacus.models.model import Model
from src.abacus.config import MAXIMUM_AR_ORDER
from src.abacus.utils.exceptions import StationarityError



class AR(Model):
    """Auto Regressive AR(p) model.

    Order choosen by PACF using a 95 % confidence interval.

    Standard deviation estamted by statistical estimators of mean and
    population standard deviation.

    Parameter estimation by Least Squares with QR decomposition. Unit roots of
    the parameters are checeked.

    Mean squared error is computed lazy with QR decomposition.

    PERMITTED INSTRUMENTS:
        * STOCK
        * FX

    NOTE:
        * Order can be ascertained through regression like t-tests (Liuns Table).
        * Model signifiance can be ascertained through regression like F-test.

    REFERENCES:
        * Time Series Analysis by Hamilton.
        * PACF Statistical Cutoff:
          http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/xegbohtmlnode39.html
    """

    def __init__(self, data: torch.Tensor):
        super().__init__(data)

    @property
    def parameters(self) -> list[float]:
        self._check_calibration()
        parameters = [value.item() for value in self._phi]
        parameters.append(self._mu.item())
        parameters.append(self._sigma.item())
        return parameters

    @property
    def _predictions(self) -> torch.Tensor:
        return self._Q @ (self._R @ self._solution)

    @property
    def _number_of_parameters(self) -> int:
        return len(self.parameters)

    @property
    def _log_likelihood(self) -> torch.Tensor:
        """
        Estimates the conditional log-likelihood of the process.

        The log likelihood assumes Gaussian innovations as outlined in [soruce].
        """
        predictions = self._predictions
        predicable_data = self._data[self._order:]
        difference = predictions - predicable_data
        squared_difference = torch.dot(difference, difference)
        variance = torch.square(self._sigma)
        pi = torch.tensor(torch.pi)

        return - (self._number_of_observations - self._order) / 2 * (torch.log(pi) + torch.log(variance)) - 1 / (2 * variance) * torch.sum(squared_difference)

    def calibrate(self):
        self._build_lag_order()
        self._build_data_matricies()
        self._solve_least_squares()
        self._estimate_standard_deviation()
        self._check_unit_roots()

        self._calibrated = True

    def transform_to_true(self, uniform_sample: torch.Tensor) -> torch.Tensor:
        self._check_calibration()

        order = self._order
        number_of_samples = len(uniform_sample)
        normals = Normal(0,1).icdf(uniform_sample)
        simulated_values = torch.zeros(order + number_of_samples)
        simulated_values[:order] = self._data[:order]
        phi = self._phi
        drift = self._mu
        sigma = self._sigma

        for i in range(number_of_samples):
            previous_values = simulated_values[i:order+i]
            simulated_value = torch.dot(phi, previous_values) + drift + sigma * normals[i]
            simulated_values[order+i] = simulated_value

        return simulated_values[self._order:]

    def transform_to_uniform(self) -> torch.Tensor:
        self._check_calibration()
        residuals = self._build_residuals()

        return Normal(0, 1).cdf(residuals)

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

    def _estimate_standard_deviation(self):
        """
        Estimates unconditional standard deviation taking into account the order of the process.
        """
        variance = 1 / (self._number_of_observations - self._order) * torch.sum(torch.square(self._data - torch.mean(self._data)))
        self._sigma = torch.sqrt(variance)

    def _solve_least_squares(self):
        solution = torch.linalg.solve(self._R, self._Q.T @ self._b)
        self._solution = solution
        self._mu = solution[0]
        self._phi = solution[1:]

    def _build_data_matricies(self):
        """
        The data matrix is build column-wise to represent regression matrix for least
        squares solver. The data matrix is QR decomposed.
        """

        self._b = self._data[self._order:]

        cols = [torch.ones(self._number_of_observations - self._order)]
        for i in range(self._order):
            col = self._data[(self._order - 1) - i: self._number_of_observations - i - 1]
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
