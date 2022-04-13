import math
import time

import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import poisson
from scipy.stats import norm
from scipy.special import gamma
from scipy.special import kv
from scipy.integrate import quad
from scipy.optimize import root_scalar


def norm_poisson_mix_pdf(x: float, mu: float, sigma: float, kappa: float, lamb: float, number_of_terms: int = 25) \
                         -> float:
    total_mix_density = 0
    for k in range(0, number_of_terms):
        total_mix_density = total_mix_density + poisson.pmf(k, lamb) * norm.pdf(x, mu, sigma*np.sqrt(1+k*kappa**2))
    return total_mix_density


def norm_poisson_mix_cdf(x: float, mu: float, sigma: float, kappa: float, lamb: float,
                         number_of_terms: int = 25, lower_limit: float = 6.) -> float:
    return quad(norm_poisson_mix_pdf, -lower_limit, x, args=(mu, sigma, kappa, lamb))[0]


def student_poisson_mix_pdf(x: float, mu: float, sigma: float, kappa: float, lamb: float, nu: float,
                            number_of_terms: int = 10) -> float:

    normalizing_constant = (gamma((nu + 1) / 2)) / (sigma * np.sqrt(np.pi * (nu - 2)) * gamma(nu / 2))

    total_mix_density = np.exp(-lamb) * (1 + (x - mu)**2/((nu - 2) * sigma ** 2)) ** ((-nu - 1)/2)

    for k in range(1, number_of_terms):
        total_mix_density = (total_mix_density
                             + poisson.pmf(k, lamb) * 1 / (kappa * sigma * np.sqrt(2*np.pi*k))
                             * quad(_student_integral, -10, 10, args=(x, k, mu, sigma, kappa, nu))[0]
                             )
    return normalizing_constant * total_mix_density


def _student_integral(s: float, x: float, k: int, mu: float, sigma: float, kappa: float, nu: float) -> float:
    evaluation = ((1 + (s - mu) ** 2 / (sigma ** 2 * (nu - 2))) ** (-(nu + 1)/2)
                  * np.exp(-0.5 * ((x-s) / (sigma * kappa * np.sqrt(k))) ** 2)
                  )
    return evaluation


def student_poisson_mix_cdf(x: float, mu: float, sigma: float, kappa: float, lamb: float, nu: float,
                            number_of_terms: int = 10, lower_limit: float = 7.) -> float:
    return quad(student_poisson_mix_pdf, -lower_limit, x, args=(mu, sigma, kappa, lamb, nu, number_of_terms))[0]


def student_poisson_mix_quantile(prob: float, mu: float, sigma: float, kappa: float, lamb: float, nu: float,
                                 number_of_terms: int = 100, lower_limit: float = 50.):
    target = prob
    sol = root_scalar(lambda x, *args: student_poisson_mix_cdf(x, *args) - target,
                      bracket=(-5 * sigma + mu, 5 * sigma + mu),
                      args=(0, 1, 1, 0.5, 5)
                      )
    return sol


def generalized_hyperbolic_poisson_mix_pdf(x: float, mu: float, sigma: float, kappa: float, lamb: float, zeta1: float,
                                           zeta2: float, zeta3: float, zeta4: float, zeta5: float,
                                           number_of_terms: int = 100) -> float:

    normalizing_constant = (((np.sqrt(zeta2**2-zeta3**2)/zeta4)**zeta1)
                            / (np.sqrt(2*np.pi) * kv(zeta1, zeta4 * np.sqrt(zeta2**2 - zeta3**2)))
                            )

    total_mix_density = (np.exp(-lamb) * np.exp(zeta4 * ((x - mu) / sigma - zeta3))
                         * (kv(zeta1-0.5, np.sqrt(zeta4**2 + ((x - mu) / sigma - zeta5)**2)))
                         / ((np.sqrt(zeta4**2 + ((x - mu) / sigma - zeta5)**2)/zeta2) ** (0.5 - zeta1))
                         )
    for k in range(1, number_of_terms):
        total_mix_density = (total_mix_density
                             + 1/(kappa * sigma * np.sqrt(k) * np.sqrt(2*np.pi)) * poisson.pmf(k, lamb)
                             * quad(_hyperbolic_integral, -100, 100,
                                    args=(x, k, mu, sigma, kappa, zeta1, zeta2, zeta4, zeta5)
                                    )[0]
                             )

    return normalizing_constant * total_mix_density


def _hyperbolic_integral(s: float, x: float, k: int,  mu: float, sigma: float, kappa: float, zeta1: float, zeta2: float,
                         zeta4: float, zeta5: float) -> float:
    evaluation = (np.exp(zeta4 * ((s - mu) / sigma - zeta5))
                  * (kv(zeta1-0.5, zeta2 * np.sqrt(zeta4**2 + ((s - mu) / sigma - zeta5)**2)))
                  / ((np.sqrt(zeta4**2 + ((s - mu) / sigma - zeta5)**2))**(0.5 - zeta1))
                  * np.exp(-0.5 * ((x - s - mu)/(kappa * sigma * np.sqrt(k)))**2)
                  )
    return evaluation


# print('__________________')
# print('0.5882155030362247')
# print(student_poisson_mix_pdf(0, 0, 1, 0.5, 0.5, 3))
# print('__________________')
# print('0.014497111303161407')
# print(student_poisson_mix_pdf(3, 0, 1, 0.5, 0.5, 3))
# print('__________________')
# print('0.4879521848980272')
# print(student_poisson_mix_pdf(3, 3, 1, 0.5, 2, 3))
# print('__________________')
# print('0.08690412342325733')
# print(student_poisson_mix_pdf(3, 6, 4, 0.5, 1.5, 3))
# print('__________________')
# print('0.032529717358629986')
# print(student_poisson_mix_pdf(3, 0, 13, 2.5, 0.5, 3))
# print('__________________')
# print('0.0001305029244456048')
# print(student_poisson_mix_pdf(3, 23, 3, 0.1, 0.1, 3))
# print('__________________')

# print(student_poisson_mix_pdf(0, 0, 1, 0.5, 0.5, 3))
# print(student_poisson_mix_pdf(3, 0, 1, 0.5, 0.5, 3))
# x = np.linspace(-10, 10, 100)
# y = np.vectorize(student_poisson_mix_cdf)
# z = y(x, 0, 1, 1, 0.5, 5)
# print(z)
# plt.plot(x, z)
# plt.show()
# plt.plot(x, norm.pdf(x, 0, 1), color='red')
# t = time.time()
# print(student_poisson_mix_quantile(0.05, 0, 1, 1, 0.5, 5))
# elapsed = time.time() - t
# print(elapsed)
# t = time.time()
# print(student_poisson_mix_quantile(0.1, 0, 1, 1, 0.5, 5))
# elapsed = time.time() - t
# print(elapsed)
# print('-0.07287438558480491')
# t = time.time()
# print(student_poisson_mix_quantile(0.90, 0, 1, 1, 0.5, 5))
# elapsed = time.time() - t
# print(elapsed)
# t = time.time()
# print(student_poisson_mix_quantile(0.95, 0, 1, 1, 0.5, 5))
# elapsed = time.time() - t
# print(elapsed)
# t = time.time()
# print(student_poisson_mix_quantile(0.99, 0, 1, 1, 0.5, 5))
# elapsed = time.time() - t
# print(elapsed)
# plt.show()

# params = {1: (0, 1, 0.09, 0.05, 4),
#           2: (0.1, 0.50, 0.35, 0.15, 5),
#           3: (-0.3, 0.75, 0.55, 0.55, 10),
#           4: (0.05, 1.1, 0.75, 0.75, 16),
#           5: (0, 0.10, 0.11, 0.15, 5)}
# lower_limit = [1, 2, 3, 4, 5]
#
# for i in params.keys():
#     t = time.time()
#     corr = val = quad(student_poisson_mix_pdf, -25, 1, args=params[i])[0]
#     elapsed = time.time() - t
#     print(f'time elapsed corr: {elapsed}')
#     for limit in lower_limit:
#         t = time.time()
#         val = quad(student_poisson_mix_pdf, -limit, 1, args=params[i])[0]
#         elapsed = time.time() - t
#         err = abs(val-corr)
#         print(f'Lower limit: {limit} yields value: {val} with error: {err} with time elapsed {elapsed}')

# print(quad(student_poisson_mix_pdf, -700, 700, args=(0, 0.50, 0.25, 0.05, 9)))
# print(quad(generalized_hyperbolic_poisson_mix_pdf, -10, 10, args=(0, 1, 0.5, 0.5, 0.1, 1, 4, 1, 0)))





