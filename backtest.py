# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

from tests.test_config import TEST_YAHOO_STOCK_UNIVERSE_16, TEST_YAHOO_STOCK_UNIVERSE_8, TEST_YAHOO_STOCK_UNIVERSE_4
from src.abacus.utils.portfolio import Portfolio
from src.abacus.utils.universe import Universe
from src.abacus.simulator.simulator import Simulator
from src.abacus.optimizer.optimizer import MPCMaximumReturn



instrument_specification = {}
inital_weights = {}
wealth = []
number_of_start_assets = 5
for i, ticker in enumerate(sorted(TEST_YAHOO_STOCK_UNIVERSE_8)):
    file = f"tests/data/{ticker}.csv"
    time_series = pd.read_csv(file, index_col='Date')
    instrument_specification[ticker] = time_series
    if i < number_of_start_assets: inital_weights[ticker] = 1 / number_of_start_assets

universe = Universe(instrument_specifications=instrument_specification)
portfolio = Portfolio(weights=inital_weights)
simulator = Simulator(universe)
simulator.calibrate()
simulator.run_simulation(time_steps=5, number_of_simulations=100)

# Date range for backtesting.
start_date = "2020-01-02"
end_date = "2020-01-03" # "2023-05-31"
date_range = pd.date_range(start=start_date, end=end_date, freq='B')

for date in date_range:
    universe.date_today = date
    simulator = Simulator(universe)
    simulator.calibrate()
    simulator.run_simulation(time_steps=5, number_of_simulations=25)
    optimizer = MPCMaximumReturn(universe, portfolio, simulator.return_tensor, gamma=1, l1_penalty=0, l2_penalty=0.05, covariance_matrix=simulator.covariance_matrix)
    optimizer.solve()

    print(optimizer.solution)
    break
