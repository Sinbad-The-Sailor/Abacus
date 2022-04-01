# An equity model can be thought of BEING a model of a STOCK itself.
# The stock will have some price history, ticker, bid-ask spreads, etc, currency, ISIN
# This data will primarily exist in the StockData class, which will be fed into the StockModel for further usage

# When creating an equity model you will feed it data from a stock_data object.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.optimize import minimize


class StockData:
    # List of data in ascending order! Since dicts are hash-tables one might think about using dicts with
    # dates as a key

    def __init__(self, ric):
        self.adj_close = pd.DataFrame()
        self.ric = ric

    def get_log_returns(self):
        return np.log(self.adj_close / self.adj_close.shift(1))[1:]


class EquityModel:
    _stock_data = None
    _model_solution = None
    _model_fitted = False

    def __init__(self, stock_data):
        self._stock_data = stock_data

    def run_simulation(self):
        pass

    def fit_model(self, model='normal'):
        data = self._stock_data.get_log_returns()
        data = data.to_list()

        if model == 'normal':
            cons = self._likelihood_constraints_normal()
            func = self._likelihood_function_normal

            # Initial conditions for GJR-GARCH(1,1) model.
            omega0 = 0.001
            alpha0 = 0.05
            beta0 = 0.80
            beta1 = 0.02
            mu0 = np.mean(data)
            x0 = [omega0, alpha0, beta0, beta1, mu0]

            garch_model_solution = minimize(func, x0, constraints=cons, args=data)

            # Added to keep method non-static. Might be useful to keep in Equity Model object instead of re-running
            # the optimization.
            self._model_solution = garch_model_solution
            self._model_fitted = True

            return garch_model_solution

        elif model == 'normal poisson mixture':
            pass

        elif model == 'generalized hyperbolic':
            cons = self._likelihood_constraints_generalized_hyperbolic()
            func = self._likelihood_constraints_generalized_hyperbolic

    # noinspection PyMethodMayBeStatic
    def _likelihood_function_normal(self, params, data) -> float:
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        # Note: vol_estimate is squared in this function.
        # Note: The negative log likelihood value is returned to be minimized instead of maximized.
        n_observations = len(data)
        log_likelihood = 0
        initial_squared_vol_estimate = (params[0]
                                        + params[1] * (data[0] ** 2)
                                        + params[3] * (data[0] ** 2) * np.where(data[0] < 0, 1, 0)
                                        + params[2] * (data[0] ** 2))
        current_squared_vol_estimate = initial_squared_vol_estimate

        for i in range(1, n_observations):
            log_likelihood = (log_likelihood
                              + ((data[i-1] - params[4]) ** 2) / current_squared_vol_estimate
                              + 2 * np.log(np.sqrt(current_squared_vol_estimate)))

            current_squared_vol_estimate = (params[0] + params[1] * (data[i-1] ** 2)
                                            + params[3] * (data[i-1] ** 2) * np.where(data[i-1] < 0, 1, 0)
                                            + params[2] * current_squared_vol_estimate)

        return log_likelihood

    # noinspection PyMethodMayBeStatic
    def _likelihood_constraints_normal(self) -> dict:
        # x[0] is omega
        # x[1] is alpha
        # x[2] is beta0
        # x[3] is beta1 (asymmetry modifier)

        cons_garch = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] - (0.5 * x[3]) + 1},
                      {'type': 'ineq', 'fun': lambda x:  x[0]},
                      {'type': 'ineq', 'fun': lambda x:  x[1] + x[3]},
                      {'type': 'ineq', 'fun': lambda x:  x[2]}]
        return cons_garch

    def _likelihood_function_normal_poisson_mixture(self, params, data):
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        n_observations = len(data)
        likelihood = 0

        # Add sum of log likelihood by PDF through normal_poisson_mixture PDF here.

    def _likelihood_function_generalized_hyperbolic(self, params, data):
        pass

    def _likelihood_constraints_generalized_hyperbolic(self):
        pass

    def plot_volatility(self):
        # Plotting GJR-GARCH fitted volatility.
        params = self._model_solution.x
        omg = params[0]
        alp = params[1]
        bet = params[2]
        gam = params[3]

        data = self._stock_data.get_log_returns()
        time = data.index[1:]
        data = data.to_list()
        vol = []

        curr_vol = (omg
                    + alp * (data[0] ** 2)
                    + gam * (data[0] ** 2) * np.where(data[0], 1, 0)
                    + bet * (data[0] ** 2))
        vol.append(curr_vol)

        for i in range(2, len(data)):
            curr_vol = (omg
                        + alp * (data[i - 1] ** 2)
                        + gam * (data[i - 1] ** 2) * np.where(data[i - 1] < 0, 1, 0)
                        + bet * (vol[i - 2]))
            vol.append(curr_vol)

        vol = np.sqrt(vol)
        plt.title(self._stock_data.ric)
        plt.plot(time[20:], vol[20:])
        plt.show()
