# -*- coding: utf-8 -*-
import numpy as np

from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
from scipy.optimize import minimize

from abacus_utils.norm_poisson_mixture import npm
from abacus_utils.student_poisson_mixture import spm

from abacus.config import ABACUS_DATABASE_CONNECTION


class EquityModel:
    def __init__(self, stock_data):
        self._stock_data = stock_data
        self._model_solution = None
        self._model_fitted = False
        self._uniform_transformed_returns = []

    def run_simulation(self):
        pass

    def fit_model(self, model='normal'):
        data = self._stock_data.get_log_returns()
        data = data['close']
        print(np.mean(data))

        if model == 'normal':
            cons = self._likelihood_constraints_normal()
            func = self._likelihood_function_normal

            # Initial conditions for GJR-GARCH(1,1) model.
            omega0 = 0.001
            alpha0 = 0.05
            beta0 = 0.80
            beta1 = 0.02
            mu0 = np.mean(data)
            x0 = np.array([omega0, alpha0, beta0, beta1, mu0])

            garch_model_solution = minimize(
                func, x0, constraints=cons, args=data)

            # Added to keep method non-static. Might be useful to keep in Equity Model object instead of re-running
            # the optimization.
            self._model_solution = garch_model_solution
            self._model_fitted = True

            return garch_model_solution

        elif model == 'normal poisson mixture':
            cons = self._likelihood_constraints_normal_poisson_mix()
            func = self._likelihood_function_normal_poisson_mixture
            init = self._init
            x0 = np.array(init)
            x0 = x0[0:-1]
            garch_poisson_model_solution = minimize(
                func, x0, constraints=cons, args=data)

            self._model_solution = garch_poisson_model_solution.x
            self._model_fitted = True

            # sol = list(self._model_solution)
            # sol.append(5)
            # database_parser.write_final_solution(ABACUS_DATABASE_CONNECTION, sol, self._stock_data.ric)

            return garch_poisson_model_solution

        elif model == 'student poisson mixture':
            # param[0] is omega
            # param[1] is alpha
            # param[2] is beta0
            # param[3] is beta1 (asymmetry modifier)
            # param[4] is mu
            # param[5] is kappa
            # param[6] is lambda
            # param[7] is nu
            # TODO: Change the order in database such that it matches with init, and the output solution.
            func = self._likelihood_function_student_poisson_mixture
            cons = self._likelihood_constraints_student_poisson_mix()
            init = self._init

            x0 = np.array(init)

            garch_poisson_model_solution = minimize(
                func, x0, constraints=cons, args=data)

            self._model_solution = garch_poisson_model_solution
            self._model_fitted = True

            return garch_poisson_model_solution

    # noinspection PyMethodMayBeStatic
    def _likelihood_function_normal(self, params, data) -> float:
        """
        Conditional GJR-GARCH likelihood function with normal innovations, derived as in John C. Hull Risk Management.
        Note, the vol_estimate is squared to simplify calculations. Note, the negative log likelihood is return to be
        minimized.

        :param params: list of parameters involved. By standard notation it follows:
                       param[0] is omega |
                       param[1] is alpha |
                       param[2] is beta0 |
                       param[3] is beta1 (asymmetry modifier) |
                       param[4] is mu
        :param data: list of observations, e.g. log-returns over time.
        :return: Negative log-likelihood value.
        """
        n_observations = len(data)
        log_likelihood = 0
        initial_squared_vol_estimate = (params[0]
                                        + params[1] * (data[0] ** 2)
                                        + params[3] * (data[0] ** 2) *
                                        np.where(data[0] < 0, 1, 0)
                                        + params[2] * (data[0] ** 2))
        current_squared_vol_estimate = initial_squared_vol_estimate

        for i in range(1, n_observations):
            log_likelihood = (log_likelihood
                              + ((data[i-1] - params[4]) ** 2) /
                              current_squared_vol_estimate
                              + 2 * np.log(np.sqrt(current_squared_vol_estimate)))

            current_squared_vol_estimate = (params[0] + params[1] * (data[i-1] ** 2)
                                            + params[3] * (data[i-1] ** 2) * np.where(data[i-1] < 0, 1, 0)
                                            + params[2] * current_squared_vol_estimate)

        return log_likelihood

    # noinspection PyMethodMayBeStatic
    def _likelihood_constraints_normal(self) -> dict:
        """

        :return:
        """
        # x[0] is omega
        # x[1] is alpha
        # x[2] is beta0
        # x[3] is beta1 (asymmetry modifier)

        cons_garch = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] - (0.5 * x[3]) + 1},
                      {'type': 'ineq', 'fun': lambda x:  x[0]},
                      {'type': 'ineq', 'fun': lambda x:  x[1] + x[3]},
                      {'type': 'ineq', 'fun': lambda x:  x[2]}]
        return cons_garch

    # noinspection PyMethodMayBeStatic
    def _likelihood_function_normal_poisson_mixture(self, params, data):
        """

        :param params:
        :param data:
        :return:
        """
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        # param[5] is kappa
        # param[6] is lambda
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
                                            + params[3] * (data[i - 1] ** 2) * np.where(data[i - 1] < 0, 1, 0)
                                            + params[2] * current_squared_vol_estimate)

        return -log_likelihood

    # noinspection PyMethodMayBeStatic
    def _likelihood_constraints_normal_poisson_mix(self) -> dict:
        cons_garch_poisson = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] - (0.5 * x[3]) + 1},
                              {'type': 'ineq', 'fun': lambda x: x[0]},
                              {'type': 'ineq', 'fun': lambda x: x[1] + x[3]},
                              {'type': 'ineq', 'fun': lambda x: x[2]},
                              {'type': 'ineq', 'fun': lambda x: x[5]},
                              {'type': 'ineq', 'fun': lambda x: x[6]}
                              ]
        return cons_garch_poisson

    # noinspection PyMethodMayBeStatic
    def _likelihood_function_student_poisson_mixture(self, params, data):
        """

        :param params:
        :param data:
        :return:
        """
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        # param[5] is kappa
        # param[6] is lambda
        # param[7] is nu
        n_observations = len(data)
        log_likelihood = 0
        initial_squared_vol_estimate = (params[0]
                                        + params[1] * (data[0] ** 2)
                                        + params[3] * (data[0] ** 2) *
                                        np.where(data[0] < 0, 1, 0)
                                        + params[2] * (data[0] ** 2))
        current_squared_vol_estimate = initial_squared_vol_estimate

        for i in range(0, n_observations):
            log_likelihood = log_likelihood + np.log(spm.pdf(data[i], params[4],
                                                     np.sqrt(
                                                         current_squared_vol_estimate),
                                                     params[5], params[6], params[7]
                                                     ))

            current_squared_vol_estimate = (params[0] + params[1] * (data[i - 1] ** 2)
                                            + params[3] * (data[i - 1] ** 2) * np.where(data[i - 1] < 0, 1, 0)
                                            + params[2] * current_squared_vol_estimate)

        return -log_likelihood

    # noinspection PyMethodMayBeStatic
    def _likelihood_constraints_student_poisson_mix(self) -> dict:
        """

        :return:
        """
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        # param[5] is kappa
        # param[6] is lambda
        # param[7] is nu
        abstol = 1e-6
        cons_garch_poisson = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] - (0.5 * x[3]) + 1},
                              {'type': 'ineq', 'fun': lambda x: x[0]-abstol},
                              {'type': 'ineq',
                                  'fun': lambda x: x[1] + x[3]-abstol},
                              {'type': 'ineq', 'fun': lambda x: x[2]-abstol},
                              {'type': 'ineq', 'fun': lambda x: x[5]-abstol},
                              {'type': 'ineq', 'fun': lambda x: x[6]-abstol},
                              {'type': 'ineq', 'fun': lambda x: x[7]-3-abstol}]
        return cons_garch_poisson

    def plot_volatility(self):
        # Plotting GJR-GARCH fitted volatility.
        params = self._model_solution.x
        omg = params[0]
        alp = params[1]
        bet = params[2]
        gam = params[3]

        data = self._stock_data.get_log_returns()
        time = data.index[1:]
        data = data['close']
        vol = []

        curr_vol = (omg
                    + alp * (data[0] ** 2)
                    + gam * (data[0] ** 2) * np.where(data[0], 1, 0)
                    + bet * (data[0] ** 2))
        vol.append(curr_vol)

        for i in range(2, len(data)):
            curr_vol = (omg
                        + alp * (data[i - 1] ** 2)
                        + gam * (data[i - 1] ** 2) *
                        np.where(data[i - 1] < 0, 1, 0)
                        + bet * (vol[i - 2]))
            vol.append(curr_vol)

        vol = np.sqrt(vol)
        plt.title(self._stock_data.ric)
        plt.plot(data.index[21:], vol[20:])
        plt.show()

    def _generate_uniform_return_observations(self):
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        # param[5] is kappa
        # param[6] is lambda
        # param[7] is nu

        uniforms = []
        data = self._stock_data.get_log_returns()
        data = data['close']
        params = self._model_solution.x
        curr_vol = (params[0]
                    + params[1] * (data[0] ** 2)
                    + params[3] * (data[0] ** 2) * np.where(data[0], 1, 0)
                    + params[2] * (data[0] ** 2))

        for i in range(1, len(data)):
            # Student T Mixture
            u = spm.cdf(data[i], params[4], np.sqrt(curr_vol),
                        params[5], params[6], params[7])
            # Normal Mixture
            # u = npm.cdf(data[i], params[4], np.sqrt(curr_vol), params[5], params[6])
            print(u)
            u = norm.ppf(u, 0, 1)

            # When using normal, apply this to find quantile-quantile plot.
            # u = (data[i] - params[4]) / np.sqrt(curr_vol)
            # u = norm.cdf(u, 0, 1)
            uniforms.append(u)
            curr_vol = (params[0]
                        + params[1] * (data[i] ** 2)
                        + params[3] * (data[i] ** 2) * np.where(data[i], 1, 0)
                        + params[2] * (curr_vol ** 2))
            print(len(uniforms))
        return np.array(uniforms)

    def plot_qq(self):
        uniforms = self._generate_uniform_return_observations()
        qqplot(uniforms, norm, fit=False, line='q')
        plt.title(self._stock_data.ric)
        plt.show()
