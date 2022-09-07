# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm
from abacus.simulator.model import Model


class NNAR(Model):
    # TODO: MLE estimation of sigma.
    # TODO: MLE estimation of mu.
    def __init__(self, data, p):
        super().__init__(data)
        self.p = p
        self.net = _Net(p)

    @property
    def initial_solution(self) -> np.array:
        """
        Not required for NNAR(p) model.
        """
        pass

    @property
    def mse(self) -> float:
        number_of_observations = len(self.data) - self.p
        residuals = self._generate_residuals()
        return np.sum(residuals ** 2) / number_of_observations

    def fit_model(self):
        """
        Fits the feed forward neural network. The parameter mu is assumed to follow from the neural network.
        In addition, the standard devation is estimated as the unconditional standard deviation of the data.
        """
        lag = self.p
        number_of_observations = len(self.data)

        input_data = []
        for i in range(lag):
            input_data.append(self.data[lag-i-1:number_of_observations-i-1])
        input_data = np.stack(input_data).T
        output_data = self.data[lag:]

        input_data = torch.Tensor(input_data)
        output_data = torch.Tensor(output_data)

        optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        for i in range(number_of_observations-lag):
            X = input_data[i,:lag]
            y = output_data[i].view(1)
            self.net.zero_grad()
            output = self.net.forward(X)
            loss = self.net.loss(output, y)
            loss.backward()
            optimizer.step()

        self.net.eval()
        self.solution = np.std(self.data)

    def _cost_function(self) -> float:
        """
        Not required for NNAR(p) model.
        """
        pass

    def run_simulation(self, number_of_steps: int) -> np.array:
        """
        Runs univariate simulation of process.

        Args:
            number_of_steps (int): number of simulation steps into the future.

        Returns:
            np.array: simulated process.
        """
        simulated_process = np.zeros(number_of_steps)
        current_regression_values = self.data[-self.p :]
        sigma = self.solution

        for i in range(number_of_steps):
            residual = np.random.normal()
            simulated_process[i] = (
                self.net(torch.Tensor(current_regression_values)) + sigma * residual
            )
            current_regression_values = np.insert(
                current_regression_values[:-1], 0, simulated_process[i]
            )

        return simulated_process

    def transform_to_true(self, uniform_sample: np.array) -> np.array:
        """
        Transforms a predicted uniform sample to true values of the process. Very similar to the
        univarite simulation case, the difference is only that uniform samples are obtained from
        elsewhere.

        Args:
            uniform_sample (np.array): sample of uniform variables U(0,1).

        Returns:
            np.array: simulated process.
        """
        number_of_observations = len(uniform_sample)
        simulated_process = np.zeros(number_of_observations)
        current_regression_values = self.data[-self.p :]
        sigma = self.solution

        for i in range(number_of_observations):
            residual = norm.ppf(uniform_sample[i])
            simulated_process[i] = (
                self.net(torch.Tensor(current_regression_values)) + sigma * residual
            )
            current_regression_values = np.insert(
                current_regression_values[:-1], 0, simulated_process[i]
            )

        return simulated_process

    def transform_to_uniform(self) -> np.array:
        """
        Transformes the normalized time series to uniform variables, assuming Gaussian White Noise. Uses
        a standard normalization approach for the first p values to avoid shrinking the dataset.

        Returns:
            np.array: sample of uniform variables U(0,1).
        """
        number_of_observations = len(self.data)
        uniform_sample = np.zeros(number_of_observations)
        residuals = self._generate_residuals()
        sigma = self.solution

        for i in range(number_of_observations):
            if i <= self.p - 1:
                uniform_sample[i] = norm.cdf((self.data[i] - np.mean(self.data)) / sigma)
            else:
                uniform_sample[i] = norm.cdf(
                    (self.data[i] - residuals[i-self.p]) / sigma
                )

        return uniform_sample

    def _generate_residuals(self) -> np.array:
        """
        Helper method to recursivley generate residuals based on some set of values for params.

        Args:
            params (np.array): parameters of the model.

        Returns:
            np.array: residuals calculated based of the guessed parameters.
        """
        number_of_observations = len(self.data)
        residuals = np.zeros(number_of_observations-self.p)
        current_regression_values = self.data[:self.p]

        for i in range(number_of_observations-self.p):
            residuals[i] = self.data[i] - self.net(torch.Tensor(current_regression_values))
            current_regression_values = np.insert(current_regression_values[:-1],0,self.data[i])

        return residuals


class _Net(nn.Module):
        """
        Private class representing the feed-forward neural network.
        """
        def __init__(self, p):
            super().__init__()
            self.p = p
            self.loss = nn.MSELoss()
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(p, 20, bias = False)
            self.fc4 = nn.Linear(20, 1, bias = False)
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc4(x)
            return x
