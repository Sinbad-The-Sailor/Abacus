# -*- coding: utf-8 -*-
import time
import pandas as pd

from tests.test_config import TEST_YAHOO_STOCK_UNIVERSE_16, TEST_YAHOO_STOCK_UNIVERSE_8, TEST_YAHOO_STOCK_UNIVERSE_4
from src.abacus.utils.portfolio import Portfolio
from src.abacus.utils.universe import Universe
from src.abacus.simulator.simulator import Simulator
from src.abacus.optimizer.optimizer import MPCMaximumReturn


# Backtesting setup.
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

# Date range for backtesting.
start_date = "2020-01-02"
end_date = "2020-01-05" # "2023-05-31"
date_range = pd.date_range(start=start_date, end=end_date, freq='B')
solutions = {}
times = {}

for date in date_range:
    t1 = time.time()
    universe.date_today = date
    simulator = Simulator(universe)
    simulator.calibrate()
    simulator.run_simulation(time_steps=10, number_of_simulations=1000)
    optimizer = MPCMaximumReturn(universe, portfolio, simulator.return_tensor, gamma=1, l1_penalty=0, l2_penalty=0.05,
                                 covariance_matrix=simulator.covariance_matrix)
    optimizer.solve()

    solution = optimizer.solution
    times[date] = time.time() - t1
    portfolio.weights = solution
    solutions[date] = solution

print('\n' * 10)
for date, solution in solutions.items():
    print(f"Took {times[date]} seconds.")
    for a, w in solution.items():
        print(a, w)
    print()
