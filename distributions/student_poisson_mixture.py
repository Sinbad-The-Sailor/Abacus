import numpy as np

from scipy.stats import poisson
from scipy.special import gamma
from scipy.integrate import quad
from scipy.optimize import root_scalar
from numba import jit


class StudentPoissonMixture:
    """
    Representing all functionalities tied to a Student's t Poisson mixture model. The object is instantiated in the
    file, used by importing the file and using the instance of 'spm' ('Student Poisson mixture').
    """
    NUMBER_OF_POISSON_TERMS = 25
    LOWER_INTEGRATION_LIMIT = 7.

    def pdf(self, x: float, mu: float, sigma: float, kappa: float, lamb: float, nu: float,
            number_of_terms: int = 10) -> float:
        normalizing_constant = (gamma((nu + 1) / 2)) / (sigma * np.sqrt(np.pi * (nu - 2)) * gamma(nu / 2))
        total_mix_density = np.exp(-lamb) * (1 + (x - mu) ** 2 / ((nu - 2) * sigma ** 2)) ** ((-nu - 1) / 2)

        for k in range(1, number_of_terms):
            total_mix_density = (total_mix_density
                                 + poisson.pmf(k, lamb) * 1 / (kappa * sigma * np.sqrt(2 * np.pi * k))
                                 * quad(self._student_integral, -10, 10, args=(x, k, mu, sigma, kappa, nu))[0]
                                 )
        return normalizing_constant * total_mix_density

    def cdf(self, x: float, mu: float, sigma: float, kappa: float, lamb: float, nu: float,
            number_of_terms: int = NUMBER_OF_POISSON_TERMS, lower_limit: float = LOWER_INTEGRATION_LIMIT) -> float:
        return quad(self.pdf, -lower_limit, x, args=(mu, sigma, kappa, lamb, nu, number_of_terms))[0]

    def ppf(self, prob: float, mu: float, sigma: float, kappa: float, lamb: float, nu: float,
            number_of_terms: int = 100, lower_limit: float = 50.):
        target = prob
        sol = root_scalar(lambda x, *args: self.cdf(x, *args) - target,
                          bracket=(-5 * sigma + mu, 5 * sigma + mu),
                          args=(0, 1, 1, 0.5, 5)
                          )
        return sol

    @staticmethod
    def _student_integral(s: float, x: float, k: int, mu: float, sigma: float, kappa: float, nu: float) -> float:
        evaluation = ((1 + (s - mu) ** 2 / (sigma ** 2 * (nu - 2))) ** (-(nu + 1) / 2)
                      * np.exp(-0.5 * ((x - s) / (sigma * kappa * np.sqrt(k))) ** 2)
                      )
        return evaluation


spm = StudentPoissonMixture()
