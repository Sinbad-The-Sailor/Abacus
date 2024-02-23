# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

from datetime import datetime as dt
from matplotlib import pyplot
from tests.test_config import TEST_YAHOO_STOCK_UNIVERSE_16, TEST_YAHOO_STOCK_UNIVERSE_8, TEST_YAHOO_STOCK_UNIVERSE_4
from src.abacus.utils.instrument import Instrument
from src.abacus.utils.portfolio import Portfolio
from src.abacus.simulator.simulator import Simulator
from src.abacus.assessor.risk_assessor import RiskAssessor
from src.abacus.optimizer.optimizer import SPMaximumUtility, MPCMaximumUtility, MPCMaximumReturn


"""
Starting 2020, monthly updates to 2023.
Record portfolio weights
Record portfolio returns
Record 1/n returns
"""

time_series_data = {}
instruments = []
initial_weights = {}
inital_holdings = {}
inital_cash = 10_000

# TODO: Should be in a universe class maybe...
instrument_mapping = {}
for id, ticker in enumerate(sorted(TEST_YAHOO_STOCK_UNIVERSE_8)):
    file = f"tests/data/{ticker}.csv"
    time_series = pd.read_csv(file, index_col='Date')
    time_series_data[ticker] = time_series
    ins = Instrument(id, ticker, "Stock", None)
    instruments.append(ins)
    initial_weights[ins] = 1 / len(TEST_YAHOO_STOCK_UNIVERSE_8)
    inital_holdings[ins] = 10
    instrument_mapping[ticker] = ins

ts = time_series_data["XOM"]
start_date = "2020-01-02"
end_date = "2020-01-03" # "2023-05-31"
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
pr = pd.date_range(start=start_date, end=end_date, freq='B')

portfolio1 = Portfolio(weights=initial_weights)
portfolio2 = Portfolio(weights=initial_weights)
portfolio3 = Portfolio(holdings=inital_holdings, cash=inital_cash)


wealth = np.zeros(len(pr))
for i, date in enumerate(pr):


    # Build universe.
    for ins in instruments:
        ins.price_history = time_series_data[ins.identifier].loc[:str(date)]

    # Build simulator.
    simulator = Simulator(instruments)
    simulator.calibrate()
    simulator.run_simulation(time_steps=5, number_of_simulations=25)

    # Run optimizer on portfolio.
    optimizer = MPCMaximumReturn(portfolio1, simulator.return_tensor, gamma=10, l1_penalty=0, l2_penalty=1, covariance_matrix=simulator.covariance_matrix)
    optimizer.solve()
    solution = optimizer.solution
    solution = {instrument_mapping[ticker]: weight for ticker, weight in solution.items()}
    print(solution)
    # optimizer = MPCMaximumUtility(portfolio2, simulator.return_tensor, gamma=1)
    # optimizer.solve()

    # optimizer = SPMaximumUtility(portfolio3, simulator.price_tensor, simulator._inital_prices, gamma=-3)
    # optimizer.solve()

    exit()


    # Update portfolio weights.

    # Record portfolio wealth.
    wealth[i] = 0
