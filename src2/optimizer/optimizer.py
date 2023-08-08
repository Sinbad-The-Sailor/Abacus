# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd

from amplpy import AMPL, Environment

from .enums import OptimizationModels
from utils.config import DEFAULT_SOLVER



class Optimizer:

    def __init__(self, optimization_model: OptimizationModels, simulation_tensor: torch.Tensor, solver: str = DEFAULT_SOLVER):
        self._optimization_model = optimization_model
        self._simulation_tensor = simulation_tensor
        self._solver = solver
        self._ran = False

    def run(self):
        self._initiate_ampl_engine()
        self._set_ampl_data()
        self._solve_optimzation_problem()
        print(self._ampl.get_variable("x_buy").get_values())
        print(self._ampl.get_variable("x_sell").get_values())
        self._ran = True

    @property
    def solution(self):
        self._check_ran()
        ...

    def _initiate_ampl_engine(self):
        environment = Environment(os.environ.get("AMPL_PATH"))
        self._ampl = AMPL(environment)
        self._ampl.option["solver"] = self._solver
        self._ampl.read(f"optimization_models/{self._optimization_model.value}")

    def _set_ampl_data(self):
        if self._optimization_model == OptimizationModels.SP_MAXIMIZE_UTILITY:
            price_tensor = np.array(self._simulation_tensor[:,-1,:])
            tensor_size = price_tensor.shape
            number_of_assets = tensor_size[0]
            number_of_scenarios = tensor_size[1]

            # TODO: Understand this. Possibly change to set of assets instead.
            price_dict = {(j, i): price_tensor[i-1][j-1] for i in range(1, number_of_assets+1) for j in range(1, number_of_scenarios+1)}

            # TODO: Remove this. Should be replaced with list[str] of tickers.
            self._ampl.get_set("assets").set_values([i for i in range(1, number_of_assets+1)])

            self._ampl.param["gamma"] = -12
            self._ampl.param["risk_free_rate"] = 0.04
            self._ampl.param["dt"] = 1/365
            self._ampl.param["number_of_assets"] = number_of_assets
            self._ampl.param["number_of_scenarios"] = number_of_scenarios
            self._ampl.param["inital_cash"] = 10_000_000
            self._ampl.param["inital_holdings"] = np.zeros(number_of_assets)
            self._ampl.param["prices"] = price_dict


        elif self._optimization_model == OptimizationModels.MPC_MAXIMIZE_UTILITY:
            ...

    def _solve_optimzation_problem(self):
        self._ampl.solve()

    def _check_ran(self):
        if not self._ran:
            raise ValueError("Optimizer has not been run.")
