# -*- coding: utf-8 -*-
from cmath import log
from pyrsistent import v
from scipy.optimize import minimize
import numpy as np
from sklearn.covariance import log_likelihood
from instruments.instruments import Equity

# CREATE ASSETS.

long_run_average = np.std(data)
number_of_observations = len(data)


# GARCH FILTER
def _generate_squared_volatility(data, params):
    result = np.zeros(number_of_observations)
    for i in range(1, number_of_observations):
        if i == 1:
            vol_squared = data[i] ** 2
            result[i] = vol_squared
        else:
            vol_squared = long_run_average**2 + np.exp(-np.exp(-params[0])) * (
                np.exp(-np.exp(-params[1]))*vol_squared + (
                    1 - np.exp(-np.exp(-params[1])))*data[i-1]**2 - long_run_average**2
            )
            result[i] = vol_squared
    return result


# COST FUNCTION
def _cost_function(params, data):
    log_loss = 0
    vol_estimates = _generate_squared_volatility(data, params)

    for i in range(1, len(data)):
        log_loss -= np.log(vol_estimates[i]) + data[i] ** 2 / vol_estimates[i]

    return log_loss


alpha = 0.02
beta = 0.9

mu_corr = alpha + beta
mu_ema = beta / (alpha + beta)

z_corr = np.log(-1/(np.log(mu_corr)))
z_ema = np.log(-1/(np.log(mu_ema)))

init_params = [z_corr, z_ema]

sol = minimize(_cost_function, init_params, args=data)

new_mu_corr = np.exp(-np.exp(-sol.x[0]))
new_mu_ema = np.exp(-np.exp(-sol.x[1]))

new_beta = new_mu_corr * new_mu_ema
new_alpha = 1 - new_beta

print(new_alpha)
print(new_beta)
