# -*- coding: utf-8 -*-
import torch
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
for ticker in sorted(TEST_YAHOO_STOCK_UNIVERSE_8):
    file = f"tests/data/{ticker}.csv"
    time_series = pd.read_csv(file, index_col='Date')
    time_series_data[ticker] = time_series

ts = time_series_data["XOM"]


start_date = "2020-01-02"
end_date = "2023-05-31"
pr = pd.period_range(start=start_date, end=end_date, freq='B')


portfolio1 = Portfolio
portfolio2 = Portfolio
portfolio3 = Portfolio


for date in pr:

    # Build universe.
    instruments = []
    for id, ticker in enumerate(sorted(TEST_YAHOO_STOCK_UNIVERSE_8)):
        time_series = time_series_data["XOM"].loc[start_date:str(date)]
        ins = Instrument(id, ticker, "Stock", time_series)
        instruments.append(ins)

    # Build simulator.
    simulator = Simulator(instruments)
    simulator.calibrate()
    simulator.run_simulation(time_steps=25, number_of_simulations=25)

    # Run optimizer on portfolio.
    optimizer = MPCMaximumReturn(portfolio, simulator.return_tensor, gamma=10, l1_penalty=0, l2_penalty=1, covariance_matrix=simulator.covariance_matrix)
    optimizer.solve()




    # Update portfolio weights.

    # Record portfolio
