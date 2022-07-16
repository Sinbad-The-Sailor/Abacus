import numpy as np

from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.stats import norm
from matplotlib import pyplot as plt

EPSILON = 1e-16
SIMULATIONS = 1000


class Model(ABC):

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
    def run_simulation(self, number_of_iterations: int) -> dict:
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


# region Equity Models
class EquityModel(Model):

    def __init__(self, initial_parameters, data):
        super().__init__(initial_parameters, data)


class GARCHEquityModel(EquityModel):

    def __init__(self, initial_parameters, data):
        super().__init__(initial_parameters, data)
        self.last_volatility_estimate = 0
        self.volatility_sample = None

    def run_simulation(self, number_of_iterations: int = SIMULATIONS) -> np.array:

        result = self._generate_simulation(
            number_of_iterations=number_of_iterations, isVol=False)

        return result

    def run_volatility_simulation(self, number_of_iterations: int = SIMULATIONS) -> np.array:

        result = self._generate_simulation(
            number_of_iterations=number_of_iterations, isVol=True)

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
            number_of_iterations=number_of_observations)
        print(volatility_samples)

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

        solution = minimize(
            self._cost_function, self.initial_parameters, constraints=self._constraints(), args=self.data, method="SLSQP")
        self.optimal_parameters = solution.x
        self.last_volatility_estimate = self._generate_volatility(
            solution.x)[-1]

        if self.verbose:
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
                       {'type': 'ineq', 'fun': lambda x:  x[0] - EPSILON},
                       {'type': 'ineq', 'fun': lambda x:  x[1] - EPSILON},
                       {'type': 'ineq', 'fun': lambda x:  x[2] - EPSILON}]
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
        # TODO: MAKE THE ARRAYS SAME LENGHT! MISGUIDING OTHERWISE.
        result = np.zeros(self.number_of_observations)

        vol_est = (params[0] + params[1] *
                   (self.data[1] ** 2) + params[2] * (self.data[1] ** 2))

        result[1] = np.sqrt(vol_est)

        for i in range(2, self.number_of_observations):
            vol_est = (params[0] + params[1] *
                       (self.data[i] ** 2) + params[2] * vol_est)
            result[i] = np.sqrt(vol_est)

        return result

    def _generate_simulation(self, number_of_iterations: int, isVol: bool) -> tuple[np.array]:

        # Check if optimal parameters exist.
        if not self._has_solution():
            raise ValueError("Model has no fitted parameters.")

        # Check if initial volatility exist.
        if self.last_volatility_estimate == 0:
            raise ValueError("Model has no initial volatility estimate.")

        # Initialize empty numpy array.
        return_result = np.zeros(number_of_iterations)
        volatility_result = np.zeros(number_of_iterations)

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

            return_result[i] = return_estimate
            volatility_result[i] = volatility_estimate

        if isVol:
            return volatility_result
        else:
            return return_result

# endregion

# region FX Models


class FXModel(Model):

    def __init__(self, initial_parameters, data):
        super().__init__(initial_parameters, data)


class GARCHFXModel(EquityModel):

    def __init__(self, initial_parameters, data):
        super().__init__(initial_parameters, data)
        self.last_volatility_estimate = 0
        self.volatility_sample = None

    def run_simulation(self, number_of_iterations: int = SIMULATIONS) -> np.array:

        result = self._generate_simulation(
            number_of_iterations=number_of_iterations, isVol=False)

        return result

    def run_volatility_simulation(self, number_of_iterations: int = SIMULATIONS) -> np.array:

        result = self._generate_simulation(
            number_of_iterations=number_of_iterations, isVol=True)

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
        print(number_of_observations)
        volatility_samples = self.run_volatility_simulation(
            number_of_iterations=number_of_observations)
        print(volatility_samples)

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

        solution = minimize(
            self._cost_function, self.initial_parameters, constraints=self._constraints(), args=self.data, method="SLSQP")
        self.optimal_parameters = solution.x
        self.last_volatility_estimate = self._generate_volatility(
            solution.x)[-1]

        if self.verbose:
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
                       {'type': 'ineq', 'fun': lambda x:  x[0] - EPSILON},
                       {'type': 'ineq', 'fun': lambda x:  x[1] - EPSILON},
                       {'type': 'ineq', 'fun': lambda x:  x[2] - EPSILON}]
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
        # TODO: MAKE THE ARRAYS SAME LENGHT! MISGUIDING OTHERWISE.
        result = np.zeros(self.number_of_observations)

        vol_est = (params[0] + params[1] *
                   (self.data[1] ** 2) + params[2] * (self.data[1] ** 2))

        result[1] = np.sqrt(vol_est)

        for i in range(2, self.number_of_observations):
            vol_est = (params[0] + params[1] *
                       (self.data[i] ** 2) + params[2] * vol_est)
            result[i] = np.sqrt(vol_est)

        return result

    def _generate_simulation(self, number_of_iterations: int, isVol: bool) -> tuple[np.array]:

        # Check if optimal parameters exist.
        if not self._has_solution():
            raise ValueError("Model has no fitted parameters.")

        # Check if initial volatility exist.
        if self.last_volatility_estimate == 0:
            raise ValueError("Model has no initial volatility estimate.")

        # Initialize empty numpy array.
        return_result = np.zeros(number_of_iterations)
        volatility_result = np.zeros(number_of_iterations)

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

            return_result[i] = return_estimate
            volatility_result[i] = volatility_estimate

        if isVol:
            return volatility_result
        else:
            return return_result


# endregion
