# -*- coding: utf-8 -*-
import os
import torch
import numpy as np

from amplpy import AMPL, Environment

from .enums import OptimizationModels
from utils.config import DEFAULT_SOLVER
from utils.portfolio import Portfolio



class Optimizer:

    def __init__(self, portfolio: Portfolio, simulation_tensor: torch.Tensor, solver: str = DEFAULT_SOLVER):
        self._portfolio = portfolio
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

    @property
    def model(self):
        return self._optimization_model

    @model.setter
    def model(self, other):
        self._optimization_model = other


    def _initiate_ampl_engine(self):
        environment = Environment(os.environ.get("AMPL_PATH"))
        self._ampl = AMPL(environment)
        self._ampl.option["solver"] = self._solver
        self._ampl.read(f"optimization_models/{self._optimization_model.value}")

    def _set_ampl_data(self):
        if self._optimization_model == OptimizationModels.SP_MAXIMIZE_UTILITY:
            assets = self._portfolio.instruments
            asset_identifiers = [instrument.identifier for instrument in assets]
            instrument_holdings = np.array(list(self._portfolio.holdings.values()))
            price_tensor = np.array(self._simulation_tensor[:,-1,:])
            tensor_size = price_tensor.shape
            number_of_assets = tensor_size[0]
            number_of_scenarios = tensor_size[1]

            price_dict = {(j+1, asset.identifier): price_tensor[asset.id][j] for asset in assets
                                                                             for j in range(number_of_scenarios)}

            self._ampl.get_set("assets").set_values(asset_identifiers)
            self._ampl.param["gamma"] = -24
            self._ampl.param["risk_free_rate"] = 0.04
            self._ampl.param["dt"] = 1/365
            self._ampl.param["number_of_assets"] = number_of_assets
            self._ampl.param["number_of_scenarios"] = number_of_scenarios
            self._ampl.param["inital_cash"] = self._portfolio._cash
            self._ampl.param["inital_holdings"] = instrument_holdings
            self._ampl.param["prices"] = price_dict


        elif self._optimization_model == OptimizationModels.MPC_MAXIMIZE_UTILITY:
            ...

    def _solve_optimzation_problem(self):
        self._ampl.solve()

    def _check_ran(self):
        if not self._ran:
            raise ValueError("Optimizer has not been run.")
