# -*- coding: utf-8 -*-
import os

import torch
import numpy as np
import amplpy as ap
from abc import ABC, abstractmethod
from typing import ClassVar

from src.abacus.utils.portfolio import Portfolio
from src.abacus.utils.universe import Universe
from src.abacus.config import DEFAULT_SOLVER
from src.abacus.utils.enumerations import OptimizationSpecifications



class OptimizationModel(ABC):

    _model_specification: ClassVar[str]

    def __init__(self, universe: Universe, portfolio: Portfolio, simulation_tensor: torch.Tensor, solver: str=DEFAULT_SOLVER, verbose: bool=True):
        self._portfolio = portfolio
        self._simulation_tensor = simulation_tensor
        self._universe = universe
        self._asset_identifiers = universe.instrument_identifiers
        self._solver = solver
        self._solved = False
        self._ampl = None
        self._verbose = verbose

    def solve(self):
        # TODO: Consider a verbose mode to display command line output.
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
        self._ampl.solve(verbose=self._verbose)

    def _check_solved(self):
        if not self._solved:
            raise ValueError("Optimizer has not been run.")

    def _check_initialization(self):
        if not self._ampl:
            raise ValueError("AMPL has not been initalized.")



class SPMaximumUtility(OptimizationModel):

    _model_specification = OptimizationSpecifications.SP_MAXIMIZE_UTILITY

    def __init__(self, universe: Universe, portfolio: Portfolio, price_tensor: torch.Tensor, inital_prices: torch.Tensor, gamma: float):
        super().__init__(universe, portfolio, price_tensor)
        self._inital_prices = inital_prices
        self._gamma = gamma

    def solve(self):
        super().solve()
        print(self._ampl.get_variable("x_buy").get_values())
        print(self._ampl.get_variable("x_sell").get_values())
        print(self._ampl.eval("display OBJECTIVE;"))

    def _set_ampl_data(self):
        # TODO: Make consistent parameters.
        assets = self._portfolio.instruments
        asset_identifiers = [instrument.identifier for instrument in assets]
        instrument_holdings = np.array(list(self._portfolio.holdings.values()))
        price_tensor = np.array(self._simulation_tensor[:,-1,:])
        inital_prices =  dict(zip(asset_identifiers, np.array(self._inital_prices.reshape(8))))
        tensor_size = price_tensor.shape
        number_of_assets = tensor_size[0]
        number_of_scenarios = tensor_size[1]
        price_dict = {(j+1, asset.identifier): price_tensor[asset.id][j] for asset in assets for j in range(number_of_scenarios)}

        self._ampl.get_set("assets").set_values(asset_identifiers)
        self._ampl.param["risk_free_rate"] = 0.05
        self._ampl.param["dt"] = 10/365
        self._ampl.param["gamma"] = self._gamma
        self._ampl.param["number_of_assets"] = number_of_assets
        self._ampl.param["number_of_scenarios"] = number_of_scenarios
        self._ampl.param["inital_cash"] = self._portfolio._cash
        self._ampl.param["inital_holdings"] = instrument_holdings
        self._ampl.param["inital_prices"] = inital_prices
        self._ampl.param["prices"] = price_dict



class MPCMaximumUtility(OptimizationModel):

    _model_specification = OptimizationSpecifications.MPC_MAXIMIZE_UTILITY

    def __init__(self, universe: Universe, portfolio: Portfolio, return_tensor: torch.Tensor, gamma: float):
        super().__init__(universe, portfolio, return_tensor)
        self._gamma = gamma

    @property
    def _return_expectation_tensor(self):
        return torch.mean(self._simulation_tensor, dim=2)

    def solve(self):
        super().solve()
        print(self._ampl.get_variable("weights").get_values())
        print(self._ampl.eval("display OBJECTIVE;"))

    def _set_ampl_data(self):
        # TODO: Add these as properties in superclass.
        assets = self._portfolio.instruments
        inital_weights = self._portfolio.weights
        inital_weights = dict(zip(self._asset_identifiers, inital_weights.values()))
        expected_return_tensor = np.array(self._return_expectation_tensor)
        tensor_size = expected_return_tensor.shape
        number_of_time_steps = tensor_size[1]
        return_dict = {(j+1, asset.identifier): expected_return_tensor[asset.id][j] for asset in assets for j in range(number_of_time_steps)}

        self._ampl.get_set("assets").set_values(self._asset_identifiers)
        self._ampl.param["gamma"] = self._gamma
        self._ampl.param["number_of_time_steps"] = number_of_time_steps
        self._ampl.param["inital_weights"] = inital_weights
        self._ampl.param["returns"] = return_dict


class MPCMaximumReturn(OptimizationModel):

    _model_specification = OptimizationSpecifications.MPC_MAXIMIZE_RETURN

    def __init__(self, universe: Universe, portfolio: Portfolio, simulation_tensor: torch.Tensor, gamma: float,
                 l1_penalty: float, l2_penalty: float, covariance_matrix: torch.Tensor):
        super().__init__(universe, portfolio, simulation_tensor)
        self._gamma = gamma
        self._l1_penalty = l1_penalty
        self._l2_penalty = l2_penalty
        self._covariance_matrix = covariance_matrix

    def solve(self):
        super().solve()

    @property
    def solution(self):
        self._check_solved()
        # TODO: Should be general?
        ampl_output = self._ampl.get_variable("weights").to_pandas().loc[1].to_dict()["weights.val"]
        # print(self._ampl.eval("display OBJECTIVE;"))
        return ampl_output

    @property
    def _return_expectation_tensor(self):
        return torch.mean(self._simulation_tensor, dim=2)

    @property
    def _assets(self):
        return self._universe.instruments

    @property
    def _inital_weights(self):
        portfolio_weights = self._portfolio.weights
        universe_instruments = self._universe.instrument_identifiers
        return {identifier: portfolio_weights.get(identifier, 0) for identifier in universe_instruments}

    @property
    def _instrument_returns(self):
        expected_return_array = np.array(self._return_expectation_tensor)
        return {(j+1, asset.identifier): expected_return_array[asset.id][j] for asset in self._assets
                for j in range(self._number_of_time_steps)}

    @property
    def _number_of_time_steps(self):
        return self._return_expectation_tensor.shape[1]

    @property
    def _number_of_assets(self):
        return len(self._universe.instruments)

    @property
    def _l1_penalty_array(self):
        return self._l1_penalty * np.ones(self._number_of_assets)

    @property
    def _l2_penalty_array(self):
        return self._l2_penalty * np.ones(self._number_of_assets)

    def _set_ampl_data(self):
        self._ampl.get_set("assets").set_values(self._asset_identifiers)
        self._ampl.param["gamma"] = self._gamma
        self._ampl.param["number_of_time_steps"] = self._number_of_time_steps
        self._ampl.param["inital_weights"] = self._inital_weights
        self._ampl.param["returns"] = self._instrument_returns
        self._ampl.param["covariance"] = np.array(self._covariance_matrix)
        self._ampl.param["l1_penalty"] = self._l1_penalty_array
        self._ampl.param["l2_penalty"] = self._l2_penalty_array
