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

    def __init__(self, initial_parameters, data):
        self.initial_parameters = initial_parameters
        self.optimal_parameters = None
        self.data = data
        self.number_of_observations = len(data)
        self.normalized_sample = None
        self.uniform_sample = None
        self.verbose = False

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

    def _has_solution(self):
        if self.optimal_parameters is None:
            raise ValueError(f"Model {Model} has no solution.")
        return True
