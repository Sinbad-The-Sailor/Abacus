# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

from amplpy import AMPL

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
        self._ran = True

    def _initiate_ampl_engine(self):
        self._ampl = AMPL()
        self._ampl.option["solver"] = self._solver
        self._ampl.read(f"optimization_models/{self._optimization_model.value}")

    def _set_ampl_data(self):
        if self._optimization_model == OptimizationModels.SP_MAXIMIZE_UTILITY:
            price_tensor = np.array(self._simulation_tensor[:,-1,:])
            tensor_size = price_tensor.shape
            print(tensor_size)
            number_of_assets = tensor_size[0]
            number_of_scenarios = tensor_size[1]
            print(number_of_assets, number_of_scenarios)

            set_of_assets = self._ampl.get_set('assets')
            set_of_assets.setValues([i for i in range(1, number_of_assets+1)])


            inital_holdings = np.zeros(number_of_assets)

            self._ampl.get_parameter("gamma").set(-12)
            self._ampl.get_parameter("risk_free_rate").set(0.05)
            self._ampl.get_parameter("dt").set(1/365)
            self._ampl.get_parameter("number_of_assets").set(number_of_assets)
            self._ampl.get_parameter("number_of_scenarios").set(number_of_scenarios)
            self._ampl.get_parameter("inital_cash").set(1_000_000)
            self._ampl.get_parameter("inital_holdings").setValues(inital_holdings)
            d = {(j, i): price_tensor[i-1][j-1] for i in range(1, number_of_assets+1) for j in range(1, number_of_scenarios+1)}
            self._ampl.get_parameter("prices").setValues(d)

        elif self._optimization_model == OptimizationModels.MPC_MAXIMIZE_UTILITY:
            ...

    def _solve_optimzation_problem(self):
        self._ampl.solve()
        print()
        print(self._ampl.get_variable("x_buy").get_values())
        print()
        print(self._ampl.get_variable("x_sell").get_values())
        print()
