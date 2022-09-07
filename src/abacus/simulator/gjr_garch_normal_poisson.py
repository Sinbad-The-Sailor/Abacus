# -*- coding: utf-8 -*-
import numpy as np

from abacus.simulator.model_new import Model
from abacus.utilities.norm_poisson_mixture import npm


class GJRGARCHNormalPoissonModel(Model):
    # param[0] is omega
    # param[1] is alpha
    # param[2] is beta0
    # param[3] is beta1 (asymmetry modifier)
    # param[4] is mu
    # param[5] is kappa
    # param[6] is lambda

    def __init__(self, data):
        super().__init__(data)
        self.last_volatility_estimate = 0
        self.volatility_sample = None

    @property
    def initial_solution(self) -> np.array:
        return super().initial_solution

    @property
    def mse(self) -> float:
        raise NotImplemented

    def fit_model(self, data: np.array) -> np.array:
        return super().fit_model(data)

    def run_simulation(self, number_of_steps: int) -> np.array():
        return super().run_simulation(number_of_steps)

    def transform_to_true(self) -> np.array:
        return super().transform_to_true()

    def transform_to_uniform(self) -> np.array:
        return super().transform_to_uniform()

    def _cost_function(self, params: np.array, data: np.array) -> float:
        n_observations = len(data)
        log_likelihood = 0
        initial_squared_vol_estimate = (
            params[0]
            + params[1] * (data[0] ** 2)
            + params[3] * (data[0] ** 2) * np.where(data[0] < 0, 1, 0)
            + params[2] * (data[0] ** 2)
        )
        current_squared_vol_estimate = initial_squared_vol_estimate

        for i in range(0, n_observations):
            log_likelihood = log_likelihood + np.log(
                npm.pdf(
                    data[i],
                    params[4],
                    np.sqrt(current_squared_vol_estimate),
                    params[5],
                    params[6],
                )
            )

            current_squared_vol_estimate = (
                params[0]
                + params[1] * (data[i - 1] ** 2)
                + params[3] * (data[i - 1] ** 2) * np.where(data[i - 1] < 0, 1, 0)
                + params[2] * current_squared_vol_estimate
            )

        return -log_likelihood

    def _constraints(self) -> list[dict]:
        constraints = [
            {"type": "ineq", "fun": lambda x: -x[1] - x[2] - (0.5 * x[3]) + 1},
            {"type": "ineq", "fun": lambda x: x[0]},
            {"type": "ineq", "fun": lambda x: x[1] + x[3]},
            {"type": "ineq", "fun": lambda x: x[2]},
            {"type": "ineq", "fun": lambda x: x[5]},
            {"type": "ineq", "fun": lambda x: x[6]},
        ]
        return constraints
