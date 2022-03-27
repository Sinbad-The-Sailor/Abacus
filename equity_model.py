# An equity model can be thought of BEING a model of a STOCK itself.
# The stock will have some price history, ticker, bid-ask spreads, etc, currency, ISIN
# This data will primarily exist in the StockData class, which will be fed into the StockModel for further usage

# When creating an equity model you will feed it data from a stock_data object.
import numpy as np
import pandas as pd

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

    def __init__(self, stock_data):
        self.stock_data = stock_data

    def run_simulation(self):
        pass

    def fit_model(self, model='normal'):
        data = self.stock_data.get_log_returns()
        data = data.to_list()

        if model == 'normal':
            cons = self._likelihood_constraints_normal()
            func = self._likelihood_function_normal

            omega0 = 0.01
            alpha0 = 0.05
            beta0 = 0.90
            mu0 = np.mean(data)
            x0 = [omega0, alpha0, beta0, mu0]

            garch_model_params = minimize(func, x0, constraints=cons, args=data)

            return garch_model_params

        elif model == 'generalized hyperbolic':
            cons = self._likelihood_constraints_generalized_hyperbolic()
            func = self._likelihood_constraints_generalized_hyperbolic

            omega0 = 0.01
            alpha0 = 0.05
            beta0 = 0.90

    def _likelihood_function_normal(self, params, data):
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta
        # param[3] is µ
        # Note: vol_estimate is squared in this function.
        n_observations = len(data)
        log_likelihood = 0
        initial_squared_vol_estimate = params[0] + params[1] * (data[0] ** 2) + params[2] * (data[0] ** 2)
        current_squared_vol_estimate = initial_squared_vol_estimate

        for i in range(1, n_observations):
            log_likelihood = (log_likelihood
                              + ((data[i-1] - params[3]) ** 2) / current_squared_vol_estimate
                              + 2 * np.log(np.sqrt(current_squared_vol_estimate)))

            current_squared_vol_estimate = (params[0] + params[1] * (data[i-1] ** 2)
                                            + params[2] * current_squared_vol_estimate)

        return log_likelihood

    def _likelihood_constraints_normal(self):
        cons_garch = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] + 1},
                      {'type': 'ineq', 'fun': lambda x:  x[0]},
                      {'type': 'ineq', 'fun': lambda x:  x[1]},
                      {'type': 'ineq', 'fun': lambda x:  x[2]}]
        return cons_garch


    def _likelihood_function_generalized_hyperbolic(self, params, data):
        pass

    def _likelihood_constraints_generalized_hyperbolic(self):
        pass
