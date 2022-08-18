# -*- coding: utf-8 -*-
import numpy as np


class GeneralizedPareto:
    """
    Representing all functionalities tied to a generalized pareto distribution. The object is instantiated in the
    file, used by importing the file and using the instance of 'gp' ('generalized pareto').
    """

    @staticmethod
    def pdf(x: float, xi: float, beta: float) -> float:
        if xi == 0:
            return 1 / beta * np.exp(- x / beta)
        else:
            return 1 / beta * (1 + xi / beta * x) ** (-1 / xi - 1)

    @staticmethod
    def cdf(x: float, xi: float, beta: float) -> float:
        if xi == 0:
            return 1 - np.exp(- x / beta)
        else:
            return 1 - (1 + xi / beta * x) ** (- 1 / xi)


gp = GeneralizedPareto()
