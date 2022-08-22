# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp

from abc import ABC, abstractmethod


class MPC(ABC):
    """Return based MPC."""

    def __init__(
            self, forecast: np.array, inital_portfolio: np.array,
            cash_rate: np.array = 0.0):

        self.initial_portfolio = inital_portfolio

        self.shape = forecast.shape

        # Adding cash as an asset.
        self.number_of_assets = forecast.shape[0] + 1
        self.number_of_steps = forecast.shape[1]

        _cash_returns = np.ones(self.number_of_steps) * cash_rate
        _total_forecasts = np.vstack([_cash_returns, forecast])
        self.forecast = _total_forecasts

    @abstractmethod
    def optimize(self) -> np.array:
        pass

    @abstractmethod
    def _build_objective(self):
        pass

    @abstractmethod
    def _build_constraints(self):
        pass


class MPCDummy(MPC):

    def _build_objective(self, Z, W, R):
        objective = 0

        for t in range(self.number_of_steps):
            if t == 0:
                objective += R[:, t].T @ Z[:, t]
            else:
                objective += R[:, t].T @ (W[:, t-1] + Z[:, t])
        return objective

    def _build_constraints(self, Z, W):
        constr = []
        ones = np.ones(self.number_of_assets)

        for t in range(self.number_of_steps):
            if t == 0:
                constr += [ones.T @ Z[:, t] == 0,
                           Z[:, t] <= 1,
                           W[:, t] == self.initial_portfolio + Z[:, t],
                           W[:, t] >= 0,
                           W[:, t] <= 1]
            else:
                constr += [ones.T @ Z[:, t] == 0,
                           Z[:, t] <= 1,
                           ]
                if t < self.number_of_steps-1:
                    constr += [W[:, t] == W[:, t-1] + Z[:, t],
                               W[:, t] >= 0,
                               W[:, t] <= 1]
        return constr

    def optimize(self):
        Z = cp.Variable((self.number_of_assets, self.number_of_steps))
        W = cp.Variable((self.number_of_assets, self.number_of_steps-1))
        R = self.forecast

        objective = self._build_objective(Z, W, R)
        constr = self._build_constraints(Z, W)
        problem = cp.Problem(cp.Maximize(objective), constr)
        problem.solve()
        self.solution = [Z.value[:, 0], W.value]


class MPCLogUtility(MPC):

    def optimize(self):
        Z = cp.Variable((self.number_of_assets, self.number_of_steps))
        W = cp.Variable((self.number_of_assets, self.number_of_steps-1))
        R = self.forecast

        objective = self._build_objective(Z, W, R)
        constr = self._build_constraints(Z, W)
        problem = cp.Problem(cp.Maximize(objective), constr)
        problem.solve()
        self.solution = [Z.value[:, 0], W.value]

    def _build_objective(self, Z, W, R):
        objective = 0

        for t in range(self.number_of_steps):
            if t == 0:
                objective += R[:, t].T @ Z[:, t]
            else:
                objective += R[:, t].T @ (W[:, t-1] + Z[:, t])

        return cp.power(objective, -1.5) * (-1/1.5)

    def _build_constraints(self, Z, W):
        constr = []
        ones = np.ones(self.number_of_assets)

        for t in range(self.number_of_steps):
            if t == 0:
                constr += [ones.T @ Z[:, t] == 0,
                           Z[:, t] <= 1,
                           W[:, t] == self.initial_portfolio + Z[:, t],
                           W[:, t] >= 0,
                           W[:, t] <= 1]
            else:
                constr += [ones.T @ Z[:, t] == 0,
                           Z[:, t] <= 1,
                           ]
                if t < self.number_of_steps-1:
                    constr += [W[:, t] == W[:, t-1] + Z[:, t],
                               W[:, t] >= 0,
                               W[:, t] <= 1]
        return constr
