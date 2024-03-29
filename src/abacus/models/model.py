# -*- coding: utf-8 -*-
import torch
import pandas as pd

from abc import ABC, abstractmethod
from src.abacus.utils.exceptions import ParameterError



class Model(ABC):
    """Represents a generic discrete time series model used in Abacus. Sampling from
    a model is made using external uniform samples.
    """

    def __init__(self, data: torch.Tensor):
        self._data = data
        self._number_of_observations = len(data)
        self._calibrated = False

    @abstractmethod
    def calibrate(self):
        """Calibrates the parameters of the model. A non-calibrated model does not
        have an AIC/BIC and simulations are not possible.
        """
        pass

    @abstractmethod
    def transform_to_uniform(self):
        """Transforms the time-series samples to a uniform sample using the model implied
        distribution.

        Required for copula calibration.
        """
        pass

    @abstractmethod
    def transform_to_true(self, uniform_sample: torch.Tensor) -> torch.Tensor:
        """Transforms a uniform sample from a coupla to the simulated process by
        the model.

        Args:
            uniform_sample (torch.Tensor): Uniform sample from copula sampling.

        Returns:
            torch.Tensor: Sample of simulated process in chronological order.
        """
        pass

    @property
    @abstractmethod
    def _log_likelihood(self):
        """
        """
        pass

    @property
    @abstractmethod
    def _number_of_parameters(self):
        """
        """
        pass

    @property
    def aic(self) -> torch.Tensor:
        """
        """
        self._check_calibration()

        log_likelihood = self._log_likelihood
        number_of_parameters = torch.tensor(self._number_of_parameters)

        return 2 * (number_of_parameters - log_likelihood)

    @property
    def bic(self) -> torch.Tensor:
        """
        """
        self._check_calibration()

        log_likelihood = self._log_likelihood
        number_of_parameters = torch.tensor(self._number_of_parameters)
        number_of_observations = torch.tensor(self._number_of_observations)

        return number_of_parameters * torch.log(number_of_observations) - 2 * log_likelihood

    def _check_calibration(self):
        """
        Checks if successful calibration has been made.

        Raises:
            ParameterError: if succesful calibration was not made.
        """
        if not self._calibrated:
            raise ParameterError("Model has not been calibrated succesfully.")

    def __str__(self) -> str:
        return type(self).__name__
