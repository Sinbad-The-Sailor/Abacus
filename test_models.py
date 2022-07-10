import numpy as np

from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.stats import norm
from matplotlib import pyplot as plt

EPSILON = 1e-16


class Model(ABC):
    initial_parameters: np.array
    optimal_parameters: np.array
    uniform_transformed_samples: np.array
    data: np.array
    number_of_observations: int

    def __init__(self, initial_parameters, data):
        self.initial_parameters = initial_parameters
        self.data = data
        self.number_of_observations = len(data)

    @abstractmethod
    def fit_model(self, data: np.array) -> np.array:
        pass

    @abstractmethod
    def run_simulation(self, number_of_iterations: int) -> dict:
        pass

    @abstractmethod
    def generate_uniform_samples(self):
        pass

    @abstractmethod
    def _cost_function(self):
        pass

    def _has_solution(self):
        if self.optimal_parameters is None:
            raise ValueError(f"Model {Model} has no solution.")
        return True


class EquityModel(Model):

    def __init__(self, initial_parameters, data):
        super().__init__(initial_parameters, data)


class FXModel(Model):

    def __init__(self, initial_parameters, data):
        super().__init__(initial_parameters, data)


class GARCHEquityModel(EquityModel):

    last_volatility_estimate: float = 0
    historical_volatility_estimates = np.array

    def run_simulation(self, number_of_iterations: int) -> np.array:

        # Check if optimal parameters exist.
        if not self._has_solution():
            raise ValueError("Model has no fitted parameters.")

        # Check if initial volatility exist.
        if self.last_volatility_estimate == 0:
            raise ValueError("Model has no initial volatility estimate.")

        # Initialize empty numpy array.
        result = np.zeros(number_of_iterations)

        # Inital paramters for reursion start.
        return_estimate = self.data[-1]
        volatility_estimate = self.last_volatility_estimate

        beta0 = self.optimal_parameters[0]
        beta1 = self.optimal_parameters[1]
        beta2 = self.optimal_parameters[2]

        # Generation of return estimates.
        for i in range(number_of_iterations):
            sample = norm.rvs(size=1, loc=0, scale=1)
            volatility_estimate = np.sqrt(
                beta0 + beta1 * volatility_estimate ** 2 + beta2 * return_estimate ** 2)
            return_estimate = sample * volatility_estimate

            result[i] = return_estimate

        return result

    def generate_uniform_samples(self):
        print("Running equity uniforms.")

    def fit_model(self) -> bool:
        solution = minimize(
            self._cost_function, self.initial_parameters, constraints=self._constraints(), args=self.data, method="SLSQP")
        self.optimal_parameters = solution.x
        # TODO: REMOVE PRINT.
        print(f" {solution.x} {solution.success}")
        return solution.success

    def _cost_function(self, params: np.array, data: np.array) -> float:
        log_loss = 0
        vol_est = self._generate_volatility(params)

        for i in range(1, self.number_of_observations):
            log_loss += (np.log(vol_est[i] ** 2 + EPSILON) +
                         (data[i] ** 2)/(vol_est[i] ** 2))

        return log_loss

    def _constraints(self) -> list[dict]:
        constraints = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] + (1-EPSILON)},
                       {'type': 'ineq', 'fun': lambda x:  x[0]},
                       {'type': 'ineq', 'fun': lambda x:  x[1]},
                       {'type': 'ineq', 'fun': lambda x:  x[2]}]
        return constraints

    def plot(self):
        if not self._has_solution():
            raise ValueError("Model solution not available.")
        params = self.optimal_parameters
        vol_result = self._generate_volatility(params=params)
        plt.plot(vol_result)
        plt.show()

    def _generate_volatility(self, params: np.array) -> np.array:
        # Number of volatility observations is one less than returns.
        # Ignore first index. One observation is automatically removed.
        result = np.zeros(self.number_of_observations)

        vol_est = (params[0] + params[1] *
                   (self.data[1] ** 2) + params[2] * (self.data[1] ** 2))

        result[1] = np.sqrt(vol_est)

        for i in range(2, self.number_of_observations):
            vol_est = (params[0] + params[1] *
                       (self.data[i] ** 2) + params[2] * vol_est)
            result[i] = np.sqrt(vol_est)

        return result


class GARCHFXModel(FXModel):

    last_volatility_estimate = 0

    def run_simulation(self, number_of_iterations: int) -> np.array:

        # Check if optimal parameters exist.
        if not self._has_solution():
            raise ValueError("Model has no fitted parameters.")

        # Check if initial volatility exist.
        if self.last_volatility_estimate == 0:
            raise ValueError("Model has no initial volatility estimate.")

        # Initialize empty numpy array.
        result = np.zeros(number_of_iterations)

        # Inital paramters for reursion start.
        return_estimate = self.data[-1]
        volatility_estimate = self.last_volatility_estimate

        beta0 = self.optimal_parameters[0]
        beta1 = self.optimal_parameters[1]
        beta2 = self.optimal_parameters[2]

        # Generation of return estimates.
        for i in range(number_of_iterations):
            sample = norm.rvs(size=1, loc=0, scale=1)
            volatility_estimate = np.sqrt(
                beta0 + beta1 * volatility_estimate ** 2 + beta2 * return_estimate ** 2)
            return_estimate = sample * volatility_estimate

            result[i] = return_estimate

        return result

    def generate_uniform_samples(self):
        print("Running equity uniforms.")

    def fit_model(self) -> bool:
        solution = minimize(
            self._cost_function, self.initial_parameters, constraints=self._constraints(), args=self.data)
        self.optimal_parameters = solution.x
        return solution.success

    def _cost_function(self, params: np.array, data: np.array) -> float:
        n_observations = len(data)
        log_loss = 0
        vol_est = params[0] + params[1] * \
            (data[1] ** 2) + params[2] * (data[1] ** 2)
        for i in range(2, n_observations):
            log_loss = log_loss - \
                (np.log(vol_est + EPSILON) + (data[i] ** 2) / vol_est)
            vol_est = params[0] + params[1] * \
                (data[i] ** 2) + params[2] * vol_est

        self.last_volatility_estimate = np.sqrt(vol_est)
        print(vol_est)
        return log_loss

    def _constraints(self) -> dict:
        constraints = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] + 1},
                       {'type': 'ineq', 'fun': lambda x:  x[0]},
                       {'type': 'ineq', 'fun': lambda x:  x[1]},
                       {'type': 'ineq', 'fun': lambda x:  x[2]}]
        return constraints
