# -*- coding: utf-8 -*-
import numpy as np
import logging

from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from abacus.simulator.model import Model, NoParametersError


logger = logging.getLogger(__name__)


class GARCH(Model):
    def __init__(self, data):
        super().__init__(data)
        self.last_volatility_estimate = 0
        self.volatility_sample = None
        self.inital_volatility_esimate = np.std(self.data[:20])
        self.long_run_volatility_estimate = np.std(self.data)

    @property
    def initial_solution(self) -> np.array:
        """
        Basic initial solution for the GARCH(1,1) model with variance target.

        Returns:
            np.array: list of inital values for parameters. Formatted as: [alpha, beta].
        """
        return np.array([0.05, 0.80])

    @property
    def mse(self) -> float:
        """
        Calculates the mean squared error.

        Returns:
            float: sum of mean squared errors.
        """
        return np.sum(self.data ** 2)

    def fit_model(self) -> bool:
        """
        Fits model with trust-constr method. Nelder-Mead is also suitable for this model.

        Returns:
            np.array: optimal parameters.
        """
        initial_parameters = self._precondition_parameters(self.initial_solution)
        solution = minimize(self._cost_function, initial_parameters, args=self.data, method="trust-constr")
        self.solution = solution.x
        self.last_volatility_estimate = self._generate_volatility(
            self.solution
        )[-1]
        self.log_likelihood_value = solution.fun
        if not solution.success:
            logger.warning("minimizer not succesful.")
        return solution.success

    def _cost_function(self, params: np.array, data: np.array) -> float:
        """
        Defines the conditional log loss for the GARCH model. Calculates the loss recursively.

        Args:
            params (np.array): parameters formatted as [alpha, beta].

        Returns:
            float: log loss value.
        """
        vol_est = self._generate_volatility_squared(params)
        log_loss = np.sum(np.log(vol_est) + (data ** 2) / vol_est)
        return log_loss

    def run_simulation(self, number_of_steps: int) -> np.array:
        """
        Runs univariate simulation of process.

        Args:
            number_of_steps (int): number of simulation steps into the future.

        Returns:
            np.array: simulated process.
        """
        result = self._generate_simulation(number_of_steps=number_of_steps, isVol=False)
        return result

    def transform_to_true(self, uniform_samples: np.array) -> np.array:
        """
        Transforms a predicted uniform sample to true values of the process. Very similar to the
        univarite simulation case, the difference is only that uniform samples are obtained from
        elsewhere.

        Args:
            uniform_sample (np.array): sample of uniform variables U(0,1).

        Returns:
            np.array: simulated process.
        """
        # Create volatility samples.
        number_of_observations = len(uniform_samples)
        volatility_samples = self.run_volatility_simulation(
            number_of_steps=number_of_observations
        )

        # Initialize empty numpy array.
        result = np.zeros(number_of_observations)

        # Transform samples and unnormalize in one go.
        for i in range(0, number_of_observations):
            uniform_sample = uniform_samples[i]
            normal_sample = norm.ppf(uniform_sample, loc=0, scale=1)
            result[i] = normal_sample * volatility_samples[i]

        return result

    def transform_to_uniform(self) -> np.array:
        """
        Transformes the normalized time series to uniform variables, assuming Gaussian White Noise.

        Returns:
            np.array: sample of uniform variables U(0,1).
        """
        number_of_observations = len(self.data)
        result = np.zeros(number_of_observations - 1)

        # Check if a solution exists.
        if self.solution is None:
            raise NoParametersError

        # Check if a volatility estimate exists.
        if self.volatility_sample is None:
            self.volatility_sample = self._generate_volatility(self.solution)

        # Create normalized sample and transform it in one go.
        for i in range(1, number_of_observations):
            # TODO: REMOVE -1. Make all arrays have correct lenght.
            normalized_sample = self.data[i] / self.volatility_sample[i]
            uniform_sample = norm.cdf(normalized_sample, loc=0, scale=1)
            result[i - 1] = uniform_sample

        return result

    def run_volatility_simulation(
        self, number_of_steps: int
    ) -> np.array:
        """
        Helper method to recursivley run volatility simulation.

        Args:
            number_of_steps (int): number of simulation steps into the future.

        Returns:
            np.array: simulated volatility process.
        """
        result = self._generate_simulation(number_of_steps=number_of_steps, isVol=True)
        return result

    def plot_volatility(self):
        """
        Helper method to plot the implied volatility for optimal parameters. Helps check if
        volatility looks reasonable.

        Raises:
            NoParametersError: if the model has not been fitted.
        """
        if self.solution is None:
            raise NoParametersError("Model solution not available.")
        params = self.solution
        vol_result = self._generate_volatility(params=params)
        plt.plot(vol_result)
        plt.show()

    def _generate_volatility(self, params: np.array) -> np.array:
        """
        Helper method to recursivley calculate volatility.

        Args:
            params (np.array): parameters for the GARCH model.

        Returns:
            np.array: filtered volatility.
        """
        result = np.sqrt(self._generate_volatility_squared(params=params))
        return result

    def _generate_volatility_squared(self, params: np.array) -> np.array:
        """
        Helper method to recursivley calculate squared volatility.

        Args:
            params (np.array): parameters for the GARCH model.

        Returns:
            np.array: filtered squared volatility.
        """
        number_of_observations = len(self.data)
        result = np.zeros(number_of_observations)
        for i in range(0, number_of_observations):
            if i == 0:
                result[i] = self.inital_volatility_esimate ** 2
            else:
                mu_corr = np.exp(-np.exp(-params[0]))
                mu_ewma = np.exp(-np.exp(-params[1]))

                result[i] = self.long_run_volatility_estimate ** 2 + mu_corr * (
                    mu_ewma * result[i - 1]
                    + (1 - mu_ewma) * self.data[i - 1] ** 2
                    - self.long_run_volatility_estimate ** 2
                )
        return result

    def _generate_simulation(self, number_of_steps: int, isVol: bool) -> np.array:
        """
        Helper method to recursivley simulate either the return process or the volatility
        process.

        Args:
            params (np.array): parameters for the GARCH model.

        Returns:
            np.array: simulated return or volatility process.
        """
        # Check if optimal parameters exist.
        if self.solution is None:
            raise NoParametersError

        # Check if initial volatility exist.
        if self.last_volatility_estimate == 0:
            raise ValueError("Model has no initial volatility estimate.")

        # Initialize empty numpy array.
        return_result = np.zeros(number_of_steps)
        volatility_result = np.zeros(number_of_steps)

        # Inital paramters for reursion start.
        return_estimate = self.data[-1]
        volatility_estimate = self.last_volatility_estimate

        parameters = self._uncondition_parameters(self.solution)
        alpha = parameters[0]
        beta = parameters[1]
        omega = self.long_run_volatility_estimate ** 2 * (1 - alpha - beta)

        # Generation of return estimates.
        for i in range(number_of_steps):
            sample = norm.rvs(size=1, loc=0, scale=1)
            volatility_estimate = np.sqrt(
                omega + beta * volatility_estimate ** 2 + alpha * return_estimate ** 2
            )
            return_estimate = sample * volatility_estimate

            return_result[i] = return_estimate
            volatility_result[i] = volatility_estimate

        if isVol:
            return volatility_result
        else:
            return return_result

    @staticmethod
    def _precondition_parameters(params: np.array) -> np.array:
        """
        Preconditioning to obtain more stable optimzation problem

        Args:
            params (np.array): GARCH parameters.

        Returns:
            np.array: transformed GARCH parameters.
        """
        mu_corr = params[0] + params[1]
        mu_ewma = params[1] / (params[0] + params[1])

        z_corr = np.log(-1 / np.log(mu_corr))
        z_ewma = np.log(-1 / np.log(mu_ewma))

        return np.array([z_corr, z_ewma])

    @staticmethod
    def _uncondition_parameters(params: np.array) -> np.array:
        """
        Unconditioning to obtain more original parameters from transformed parameters.

        Args:
            params (np.array): transformed GARCH parameters.

        Returns:
            np.array: GARCH parameters.
        """
        mu_corr = np.exp(-np.exp(-params[0]))
        mu_ewma = np.exp(-np.exp(-params[1]))

        alpha = mu_corr * (1 - mu_ewma)
        beta = mu_corr * mu_ewma

        return np.array([alpha, beta])
