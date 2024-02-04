# -*- coding: utf-8 -*-
import torch


class GeneralizedPareto:
    """
    Representing all functionalities tied to a generalized pareto distribution. The object is instantiated in the
    file, used by importing the file and using the instance of 'gp' ('generalized pareto').
    """

    @staticmethod
    def pdf(x: torch.Tensor, xi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        if xi == 0:
            return 1 / beta * torch.exp(-x / beta)
        else:
            return 1 / beta * (1 + xi / beta * x) ** (-1 / xi - 1)

    @staticmethod
    def cdf(x: torch.Tensor, xi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        if xi == 0:
            return 1 - torch.exp(-x / beta)
        else:
            return 1 - (1 + xi / beta * x) ** (-1 / xi)


gp = GeneralizedPareto()
