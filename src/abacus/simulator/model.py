# -*- coding: utf-8 -*-
import numpy as np

from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract class representing a model for an instrument.

    Attributes:
        initial_parameters (np.array): Array of model inital paramters.
        optimal_parameters (np.array): Array of model optimal parameters.
        data (np.array): Dataset used for parameter estimation.
        number_of_observations (int): Number of observations in dataset.
        normalized_sample (np.array): Dataset transformed to be normalized.
        uniform_sample: (np.array: Dataset transformed into a uniform sample.
        verbose (bool): Data print and plotting for debug.
    """

    def __init__(self, data=None, initial_parameters=None):
        self.initial_parameters = initial_parameters
        self.data = data
        self.number_of_observations = len(data)
        self.optimal_parameters = None
        self.normalized_sample = None
        self.uniform_sample = None
        self.verbose = False
        self.number_of_parameters = None
        self.log_likelihood_value = None

    @abstractmethod
    def fit_model(self, data: np.array) -> np.array:
        pass

    @abstractmethod
    def run_simulation(self, number_of_steps: int) -> dict:
        pass

    @abstractmethod
    def generate_uniform_samples(self):
        pass

    @abstractmethod
    def generate_correct_samples(self):
        pass

    @abstractmethod
    def _cost_function(self):
        pass

    def aic(self) -> float:
        """
        Returns the models Akaike Information Criterion (AIC) value.

        Returns:
            float: AIC value.
        """
        return 2 * self.number_of_parameters - 2 * self.log_likelihood_value

    def set_inital_parameters(self, initial_parameters: np.array):
        """
        Setter for the model's initial parameters.

        Args:
            initial_parameters (np.array): inital parameters.
        """
        self.initial_parameters = initial_parameters

    def set_data(self, data: np.array):
        """
        Setter for the model's data, normally log-return history.

        Args:
            data (np.array): data, normally log-return history.
        """
        self.data = data

    def _has_solution(self) -> bool:
        """
        Checks if model has valid optimal solution.

        Raises:
            ValueError: if model has no solution.

        Returns:
            bool: status on solution.
        """
        if self.optimal_parameters is None:
            raise ValueError(f"Model {Model} has no solution.")
        return True


class NoParametersError(ValueError):
    pass
