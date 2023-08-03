# -*- coding: utf-8 -*-
import torch
import pandas as pd

from abc import ABC, abstractmethod
from utils.exceptions import ParameterError


class Model(ABC):
    """Represents a generic discrete time series model used in Abacus. Sampling from
    a model is made using external uniform samples.
    """

    def __init__(self, time_series: pd.Series):
        self.time_series = time_series
        self._data = torch.Tensor(time_series.values)
        self._number_of_observations = len(time_series)
        self._calibrated = False

    @abstractmethod
    def calibrate(self):
        """Calibrates the parameters of the model. A non-calibrated model does not
        have a MSE and simulations are not possible.
        """
        ...

    @abstractmethod
    def mse(self):
        """Mean squared error with respect to model time-series. Based on returns of
        the process.
        """
        ...

    @abstractmethod
    def transform_to_uniform(self):
        """Transforms the time-series samples to a uniform sample using the model implied
        distribution.

        Required for copula calibration.
        """
        ...

    @abstractmethod
    def transform_to_true(self, uniform_sample: torch.Tensor) -> torch.Tensor:
        """Transforms a uniform sample from a coupla to the simulated process by
        the model.

        Args:
            uniform_sample (torch.Tensor): Uniform sample from copula sampling.

        Returns:
            torch.Tensor: Sample of simulated process in chronological order.
        """
        ...

    def _check_calibration(self):
        """
        Checks if successful calibration has been made.

        Raises:
            ParameterError: if succesful calibration was not made.
        """
        if not self._calibrated:
            raise ParameterError("Model has not been calibrated succesfully.")
