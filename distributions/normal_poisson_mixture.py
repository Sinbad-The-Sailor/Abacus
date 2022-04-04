import numpy as np

from scipy.stats import poisson
from scipy.stats import norm
from scipy.special import gamma
from scipy.integrate import quad


def norm_poisson_mix_pdf(x: float, mu: float, sigma: float, kappa: float, lamb: float, number_of_terms: int = 100) \
                         -> float:
    total_mix_density = 0
    for k in range(0, number_of_terms):
        total_mix_density = total_mix_density + poisson.pmf(k, lamb) * norm.pdf(x, mu, sigma*np.sqrt(1+k*kappa**2))
    return total_mix_density


def student_poisson_mix_pdf(x: float, mu: float, sigma: float, kappa: float, lamb: float, nu: float,
                            number_of_terms: int = 100) -> float:
    normalizing_constant = (gamma((nu + 1) / 2)) / (np.sqrt(np.pi * (nu - 2)) * gamma(nu/2))
    total_mix_density = np.exp(-lamb) * (1 + (x - mu) ** 2 / (sigma ** 2 * (nu - 1))) ** ((-nu - 1)/2)
    for k in range(1, number_of_terms):
        total_mix_density = (total_mix_density
                             + 1/(kappa * sigma * np.sqrt(2*k*np.pi)) * poisson.pmf(k, lamb)
                             * quad(_student_integral, -100, 100, args=(x, k, mu, sigma, kappa, nu))[0]
                             )
    return normalizing_constant * total_mix_density


def _student_integral(s: float, x: float, k: int, mu: float, sigma: float, kappa: float, nu: float) -> float:
    evaluation = ((1 + s ** 2 / (sigma ** 2 * (nu - 1))) ** (-(nu + 1)/2)
                  * np.exp(-0.5 * ((x-s-mu) / (sigma * kappa * np.sqrt(k))) ** 2)
                  )
    return evaluation


print(student_poisson_mix_pdf(2, 0, 1, 0.5, 0.5, 3))
print(student_poisson_mix_pdf(0, 0, 1, 0.5, 0.5, 3))
