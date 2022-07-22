# -*- coding: utf-8 -*-
from os import stat
from tkinter import E
import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm
from matplotlib import pyplot as plt

from config import DEFALUT_STEPS, EPSILON
from models.model import Model
from distributions.norm_poisson_mixture import npm


class EquityModel(Model):

    def __init__(self, initial_parameters, data):
        super().__init__(initial_parameters, data)


# region Equity Models

class GARCHEquityModel(EquityModel):

    def __init__(self, initial_parameters, data):
        super().__init__(initial_parameters, data)
        self.last_volatility_estimate = 0
        self.volatility_sample = None
        self.inital_volatility_esimate = np.std(self.data[:19])
        self.long_run_volatility_estimate = np.std(self.data)

    def run_simulation(self, number_of_steps: int = DEFALUT_STEPS) -> np.array:

        result = self._generate_simulation(
            number_of_steps=number_of_steps, isVol=False)

        return result

    def run_volatility_simulation(self, number_of_steps: int = DEFALUT_STEPS) -> np.array:

        result = self._generate_simulation(
            number_of_steps=number_of_steps, isVol=True)

        return result

    def generate_uniform_samples(self) -> np.array:

        result = np.zeros(self.number_of_observations-1)

        # Check if a solution exists.
        if not self._has_solution():
            raise ValueError("Has no valid solution")

        # Check if a volatility estimate exists.
        if self.volatility_sample is None:
            self.volatility_sample = self._generate_volatility(
                self.optimal_parameters)

        # Create normalized sample and transform it in one go.
        for i in range(1, self.number_of_observations):
            # TODO: REMOVE -1. Make all arrays have correct lenght.
            normalized_sample = self.data[i] / self.volatility_sample[i]
            uniform_sample = norm.cdf(normalized_sample, loc=0, scale=1)
            result[i-1] = uniform_sample

        return result

    def generate_correct_samples(self, uniform_samples: np.array) -> np.array:

        # Create volatility samples.
        number_of_observations = len(uniform_samples)
        volatility_samples = self.run_volatility_simulation(
            number_of_steps=number_of_observations)

        # Initialize empty numpy array.
        result = np.zeros(number_of_observations)

        # Transform samples and unnormalize in one go.
        for i in range(0, number_of_observations):
            uniform_sample = uniform_samples[i]
            normal_sample = norm.ppf(uniform_sample, loc=0, scale=1)
            result[i] = normal_sample * volatility_samples[i]

        return result

    def fit_model(self) -> bool:
        # TODO: Add number of iterations and while loop.

        initial_parameters = self._precondition_parameters(
            self.initial_parameters)

        solution = minimize(
            self._cost_function, initial_parameters, args=self.data)
        self.optimal_parameters = solution.x
        self.last_volatility_estimate = self._generate_volatility(
            self.optimal_parameters)[-1]

        print(
            f" {self._uncondition_parameters(self.optimal_parameters)} {solution.success}")

        return solution.success

    def _cost_function(self, params: np.array, data: np.array) -> float:
        vol_est = self._generate_volatility_squared(params)
        log_loss = np.sum(np.log(vol_est[1:]) + (data[1:]**2)/vol_est[1:])
        return log_loss

    def plot_volatility(self):
        if not self._has_solution():
            raise ValueError("Model solution not available.")
        params = self.optimal_parameters
        vol_result = self._generate_volatility(params=params)
        plt.plot(vol_result[1:])
        plt.show()

    def _generate_volatility(self, params: np.array) -> np.array:
        result = np.sqrt(self._generate_volatility_squared(params=params))
        return result

    def _generate_volatility_squared(self, params: np.array) -> np.array:
        result = np.zeros(self.number_of_observations)
        for i in range(1, self.number_of_observations):
            if i == 1:
                result[i] = self.inital_volatility_esimate ** 2
            else:
                result[i] = (self.long_run_volatility_estimate**2
                             + np.exp(-np.exp(-params[0]))
                             * (np.exp(-np.exp(-params[1])) * result[i-1]
                                 + (1-np.exp(-np.exp(-params[1])))
                                 * self.data[i-1]**2
                                 - self.long_run_volatility_estimate**2
                                )
                             )
        return result

    def _generate_simulation(self, number_of_steps: int, isVol: bool) -> tuple[np.array]:

        # Check if optimal parameters exist.
        if not self._has_solution():
            raise ValueError("Model has no fitted parameters.")

        # Check if initial volatility exist.
        if self.last_volatility_estimate == 0:
            raise ValueError("Model has no initial volatility estimate.")

        # Initialize empty numpy array.
        return_result = np.zeros(number_of_steps)
        volatility_result = np.zeros(number_of_steps)

        # Inital paramters for reursion start.
        return_estimate = self.data[-1]
        volatility_estimate = self.last_volatility_estimate

        beta0 = self.optimal_parameters[0]
        beta1 = self.optimal_parameters[1]
        beta2 = self.optimal_parameters[2]

        # Generation of return estimates.
        for i in range(number_of_steps):
            sample = norm.rvs(size=1, loc=0, scale=1)
            volatility_estimate = np.sqrt(
                beta0 + beta1 * volatility_estimate ** 2 + beta2 * return_estimate ** 2)
            return_estimate = sample * volatility_estimate

            return_result[i] = return_estimate
            volatility_result[i] = volatility_estimate

        if isVol:
            return volatility_result
        else:
            return return_result

    @staticmethod
    def _precondition_parameters(params: np.array) -> np.array:
        mu_corr = params[0] + params[1]
        mu_ewma = params[1] / (params[0] + params[1])

        z_corr = np.log(-1/np.log(mu_corr))
        z_ewma = np.log(-1/np.log(mu_ewma))

        return np.array([z_corr, z_ewma])

    @staticmethod
    def _uncondition_parameters(params: np.array) -> np.array:
        mu_corr = np.exp(-np.exp(-params[0]))
        mu_ewma = np.exp(-np.exp(-params[1]))

        alpha = mu_corr * (1 - mu_ewma)
        beta = mu_corr * mu_ewma

        return np.array([alpha, beta])


class GJRGARCHNormalPoissonEquityModel(EquityModel):
    # param[0] is omega
    # param[1] is alpha
    # param[2] is beta0
    # param[3] is beta1 (asymmetry modifier)
    # param[4] is mu
    # param[5] is kappa
    # param[6] is lambda

    def __init__(self, initial_parameters, data):
        super().__init__(initial_parameters, data)
        self.last_volatility_estimate = 0
        self.volatility_sample = None

    def fit_model(self, data: np.array) -> np.array:
        return super().fit_model(data)

    def run_simulation(self, number_of_steps: int) -> dict:
        return super().run_simulation(number_of_steps)

    def generate_uniform_samples(self):
        return super().generate_uniform_samples()

    def generate_correct_samples(self):
        return super().generate_correct_samples()

    def _cost_function(self, params: np.array, data: np.array) -> float:
        n_observations = len(data)
        log_likelihood = 0
        initial_squared_vol_estimate = (params[0]
                                        + params[1] * (data[0] ** 2)
                                        + params[3] * (data[0] ** 2) *
                                        np.where(data[0] < 0, 1, 0)
                                        + params[2] * (data[0] ** 2))
        current_squared_vol_estimate = initial_squared_vol_estimate

        for i in range(0, n_observations):
            log_likelihood = log_likelihood + np.log(npm.pdf(data[i], params[4],
                                                             np.sqrt(current_squared_vol_estimate), params[5], params[6]))

            current_squared_vol_estimate = (params[0] + params[1] * (data[i - 1] ** 2)
                                            + params[3] * (data[i - 1] ** 2) *
                                            np.where(data[i - 1] < 0, 1, 0)
                                            + params[2] * current_squared_vol_estimate)

        return -log_likelihood

    def _constraints(self) -> list[dict]:
        constraints = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] - (0.5 * x[3]) + 1},
                       {'type': 'ineq', 'fun': lambda x: x[0]},
                       {'type': 'ineq', 'fun': lambda x: x[1] + x[3]},
                       {'type': 'ineq', 'fun': lambda x: x[2]},
                       {'type': 'ineq', 'fun': lambda x: x[5]},
                       {'type': 'ineq', 'fun': lambda x: x[6]}
                       ]
        return constraints

# endregion
