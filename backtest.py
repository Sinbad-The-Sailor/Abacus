# -*- coding: utf-8 -*-
import time

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from tests.test_config import TEST_YAHOO_STOCK_UNIVERSE_16, TEST_YAHOO_STOCK_UNIVERSE_8, TEST_YAHOO_STOCK_UNIVERSE_4
from src.abacus.utils.portfolio import Portfolio
from src.abacus.utils.universe import Universe
from src.abacus.simulator.simulator import Simulator
from src.abacus.optimizer.optimizer import MPCMaximumReturn


# Backtesting setup.
instrument_specification = {}
inital_weights = {}
wealth = [100]
wealth_n = [100]
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
end_date = "2020-05-02" # "2023-05-31"
date_range = pd.date_range(start=start_date, end=end_date, freq='B')
solutions = {"2020-01-01": inital_weights}
times = {}

for date in date_range:
    t1 = time.time()
    universe.date_today = date
    simulator = Simulator(universe)
    simulator.calibrate()
    simulator.run_simulation(time_steps=5, number_of_simulations=1000)
    optimizer = MPCMaximumReturn(universe, portfolio, simulator.return_tensor, gamma=2, l1_penalty=0, l2_penalty=0.02,
                                 covariance_matrix=simulator.covariance_matrix)
    optimizer.solve()
    solution = optimizer.solution
    times[date] = time.time() - t1
    portfolio.weights = solution
    solutions[date] = solution

    universe_returns = universe.todays_returns
    portfolio_weights = np.array(list(solution.values()))
    equal_weights = (1/8) * np.ones(8)
    portfolio_return = np.dot(universe_returns, portfolio_weights)
    equal_return = np.dot(universe_returns, equal_weights)
    wealth.append(wealth[-1] * (1 + portfolio_return))
    wealth_n.append(wealth_n[-1] * (1 + equal_return))

dates = ["2024-01-01"]
for date in date_range:
    dates.append(str(date)[0:10])

np.savetxt('data.csv', np.array(wealth), delimiter=',')

plt.plot(dates, wealth)
plt.plot(dates, wealth_n)
plt.show()

print('\n' * 10)
for date, solution in solutions.items():
    print(f"Took {times.get(date, -1)} seconds.")
    for a, w in solution.items():
        print(a, w)
    print()
