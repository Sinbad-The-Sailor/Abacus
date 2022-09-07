# -*- coding: utf-8 -*-
import numpy as np

from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, data: np.array) -> None:
        self.data = data
        self.solution = None

    @property
    def initial_solution(self) -> np.array:
        pass

    @property
    def mse(self) -> float:
        pass

    @abstractmethod
    def fit_model(self) -> np.array:
        pass

    @abstractmethod
    def _cost_function(self) -> float:
        pass

    @abstractmethod
    def run_simulation(self, number_of_steps: int) -> np.array:
        pass

    @abstractmethod
    def transform_to_true(self) -> np.array:
        pass

    @abstractmethod
    def transform_to_uniform(self) -> np.array:
        pass


class StationarityError(ValueError):
    pass

class NoParametersError(ValueError):
    pass
