import numpy as np

from scipy.stats import norm, poisson
from scipy.integrate import quad


class NormalPoissonMixture:
    NUMBER_OF_POISSON_TERMS = 25
    LOWER_INTEGRATION_LIMIT = 6.

    def __init__(self):
        pass

    @staticmethod
    def pdf(x: float, mu: float, sigma: float, kappa: float, lamb: float,
            number_of_terms: int = NUMBER_OF_POISSON_TERMS) -> float:
        total_mix_density = 0
        for k in range(0, number_of_terms):
            total_mix_density = total_mix_density + poisson.pmf(k, lamb) * norm.pdf(x, mu,
                                                                                    sigma * np.sqrt(1 + k * kappa ** 2))
        return total_mix_density

    def cdf(self, x: float, mu: float, sigma: float, kappa: float, lamb: float,
            number_of_terms: int = NUMBER_OF_POISSON_TERMS, lower_limit: float = LOWER_INTEGRATION_LIMIT) -> float:
        return quad(self.pdf, -lower_limit, x, args=(mu, sigma, kappa, lamb, number_of_terms))[0]

    @staticmethod
    def ppf(self):
        pass


npm = NormalPoissonMixture()
