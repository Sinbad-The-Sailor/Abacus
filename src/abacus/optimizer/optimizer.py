# -*- coding: utf-8 -*-
import os

import torch
import numpy as np
import amplpy as ap
from abc import ABC, abstractmethod
from typing import ClassVar

from src.abacus.utils.portfolio import Portfolio
from src.abacus.config import DEFAULT_SOLVER
from src.abacus.utils.enumerations import OptimizationSpecifications



class OptimizationModel(ABC):

    _model_specification: ClassVar[int]

    def __init__(self, portfolio: Portfolio, simulation_tensor: torch.Tensor, solver: str=DEFAULT_SOLVER):
        self._portfolio = portfolio
        self._simulation_tensor = simulation_tensor
        self._solver = solver
        self._solved = False
        self._ampl = None

    def solve(self):
        self._initiate_ampl_engine()
        self._set_ampl_data()
        self._solve_optimzation_problem()
        self._solved = True

    @abstractmethod
    def _set_ampl_data(self):
        ...

    def _initiate_ampl_engine(self):
        environment = ap.Environment(os.environ.get("AMPL_PATH"))
        self._ampl = ap.AMPL(environment)
        self._ampl.option["solver"] = self._solver
        self._ampl.read(f"src/abacus/optimizer/optimization_models/{self._model_specification.value}")

    def _solve_optimzation_problem(self):
        self._check_initialization()
        self._ampl.solve()

    def _check_solved(self):
        if not self._solved:
            raise ValueError("Optimizer has not been run.")

    def _check_initialization(self):
        if not self._ampl:
            raise ValueError("AMPL has not been initalized.")

    def _check_valid_model_specification(self):
        # TODO: Implement when approriate.
        ...



class SPMaximumUtility(OptimizationModel):

    _model_specification = OptimizationSpecifications.SP_MAXIMIZE_UTILITY

    def __init__(self, portfolio: Portfolio, price_tensor: torch.Tensor, inital_prices: torch.Tensor):
        super().__init__(portfolio, price_tensor)
        self._inital_prices = inital_prices

    def solve(self):
        super().solve()
        print(self._ampl.get_variable("x_buy").get_values())
        print(self._ampl.get_variable("x_sell").get_values())
        print(self._ampl.eval("display Objective;"))

    def _set_ampl_data(self):
        assets = self._portfolio.instruments
        asset_identifiers = [instrument.identifier for instrument in assets]
        instrument_holdings = np.array(list(self._portfolio.holdings.values()))
        price_tensor = np.array(self._simulation_tensor[:,-1,:])
        inital_prices =  dict(zip(asset_identifiers, np.array(self._inital_prices.reshape(4))))
        tensor_size = price_tensor.shape
        number_of_assets = tensor_size[0]
        number_of_scenarios = tensor_size[1]
        price_dict = {(j+1, asset.identifier): price_tensor[asset.id][j] for asset in assets for j in range(number_of_scenarios)}

        self._ampl.get_set("assets").set_values(asset_identifiers)
        self._ampl.param["risk_free_rate"] = 0.05
        self._ampl.param["dt"] = 10/365
        self._ampl.param["number_of_assets"] = number_of_assets
        self._ampl.param["number_of_scenarios"] = number_of_scenarios
        self._ampl.param["inital_cash"] = self._portfolio._cash
        self._ampl.param["inital_holdings"] = instrument_holdings
        self._ampl.param["inital_prices"] = inital_prices
        self._ampl.param["prices"] = price_dict


class MPCMaximumUtility(OptimizationModel):

    _model_specification = OptimizationSpecifications.MPC_MAXIMIZE_UTILITY

    def __init__(self, portfolio: Portfolio, return_tensor: torch.Tensor):
        super().__init__(portfolio, return_tensor)
        raise NotImplementedError

    def _set_ampl_data(self):
        return super()._set_ampl_data()
