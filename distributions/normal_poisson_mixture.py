import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import poisson
from scipy.stats import norm
from scipy.special import gamma
from scipy.special import kv
from scipy.integrate import quad


def norm_poisson_mix_pdf(x: float, mu: float, sigma: float, kappa: float, lamb: float, number_of_terms: int = 100) \
                         -> float:
    total_mix_density = 0
    for k in range(0, number_of_terms):
        total_mix_density = total_mix_density + poisson.pmf(k, lamb) * norm.pdf(x, mu, sigma*np.sqrt(1+k*kappa**2))
    return total_mix_density


def student_poisson_mix_pdf(x: float, mu: float, sigma: float, kappa: float, lamb: float, nu: float,
                            number_of_terms: int = 100) -> float:
    normalizing_constant = (gamma((nu + 1) / 2)) / (sigma * np.sqrt(np.pi * (nu - 2)) * gamma(nu / 2))
    total_mix_density = np.exp(-lamb) * (1 + (x-mu)**2/((nu - 2) * sigma ** 2)) ** ((-nu - 1)/2)
    for k in range(1, number_of_terms):
        total_mix_density = (total_mix_density
                             + poisson.pmf(k, lamb) * 1 / (kappa * sigma * np.sqrt(2*np.pi*k))
                             * quad(_student_integral, -100, 100, args=(x, k, mu, sigma, kappa, nu))[0]
                             )
    return normalizing_constant * total_mix_density


def _student_integral(s: float, x: float, k: int, mu: float, sigma: float, kappa: float, nu: float) -> float:
    evaluation = ((1 + (s - mu) ** 2 / (sigma ** 2 * (nu - 1))) ** (-(nu + 1)/2)
                  * np.exp(-0.5 * ((x-s) / (sigma * kappa * np.sqrt(k))) ** 2)
                  )
    return evaluation


def generalized_hyperbolic_poisson_mix_pdf(x: float, mu: float, sigma: float, kappa: float, lamb: float, zeta1: float,
                                           zeta2: float, zeta3: float, zeta4: float, zeta5: float,
                                           number_of_terms: int = 100) -> float:
    normalizing_constant = (((np.sqrt(zeta2**2-zeta3**2)/zeta4)**zeta1)
                            / (np.sqrt(2*np.pi) * kv(zeta1, zeta4 * np.sqrt(zeta2**2 - zeta3**2)))
                            )
    total_mix_density = (np.exp(-lamb) * np.exp(zeta4 * (x - zeta3))
                         * (kv(zeta1-0.5, np.sqrt(zeta4**2 + (x - zeta5)**2)))
                         / ((np.sqrt(zeta4**2 + (x-zeta5)**2)/zeta2) ** (0.5 - zeta1))
                         )
    for k in range(1, number_of_terms):

        total_mix_density = (total_mix_density
                             + 1/(kappa * sigma * np.sqrt(k)) * poisson.pmf(k, lamb)
                             * quad(_hyperbolic_integral, -100, 100,
                                    args=(x, k, mu, sigma, kappa, zeta1, zeta2, zeta4, zeta5)
                                    )[0]
                             )

    return normalizing_constant * total_mix_density


def _hyperbolic_integral(s: float, x: float, k: int,  mu: float, sigma: float, kappa: float, zeta1: float, zeta2: float,
                         zeta4: float, zeta5: float) -> float:
    evaluation = (np.exp(zeta4 * (s - zeta5))
                  * (kv(zeta1-0.5, zeta2 * np.sqrt(zeta4**2 + (s-zeta5)**2)))
                  / ((np.sqrt(zeta4**2 + (s-zeta5)**2))**(0.5 - zeta1))
                  * np.exp(-0.5 * ((x - s - mu)/(kappa * sigma * np.sqrt(k)))**2)
                  )
    return evaluation


#print(student_poisson_mix_pdf(0, 0, 1, 0.5, 0.5, 3))
#print(student_poisson_mix_pdf(3, 0, 1, 0.5, 0.5, 3))
#x = np.linspace(-10, 10, 100)
#y = np.vectorize(student_poisson_mix_pdf)
#plt.plot(x, y(x, 0, 1, 1, 0.5, 3))
#plt.plot(x, y(x, 0, 1, 0.1, 0.5, 10))
#plt.plot(x, norm.pdf(x, 0, 1))
#plt.show()

print(quad(student_poisson_mix_pdf, -50, 50, args=(0, 1, 0.5, 0.5, 5)))
#print(quad(generalized_hyperbolic_poisson_mix_pdf, -10, 10, args=(0, 1, 0.5, 0.5, 0.1, 2, 1, 1, 0)))
