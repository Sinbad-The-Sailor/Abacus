import numpy as np

from scipy.stats import poisson
from scipy.stats import norm


def norm_poisson_mix_pdf(x: float, mu: float, sigma: float, kappa: float, lamb: float, number_of_terms: int = 100) \
                         -> float:
    total_mix_density = 0
    for k in range(0, number_of_terms):
        total_mix_density = total_mix_density + poisson.pmf(k, lamb) * norm.pdf(x, mu, sigma*np.sqrt(1+k*kappa**2))
    return total_mix_density
