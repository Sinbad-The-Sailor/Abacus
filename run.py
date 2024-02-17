# -*- coding: utf-8 -*-
import torch
import pandas as pd

from matplotlib import pyplot
from tests.test_config import TEST_YAHOO_STOCK_UNIVERSE_16, TEST_YAHOO_STOCK_UNIVERSE_8, TEST_YAHOO_STOCK_UNIVERSE_4
from src.abacus.utils.instrument import Instrument
from src.abacus.utils.portfolio import Portfolio
from src.abacus.simulator.simulator import Simulator
from src.abacus.assessor.risk_assessor import RiskAssessor
from src.abacus.optimizer.optimizer import SPMaximumUtility, MPCMaximumUtility



# Mock instrument creation...
instruments = []
for id, ticker in enumerate(sorted(TEST_YAHOO_STOCK_UNIVERSE_8)):
    file = f"tests/data/{ticker}.csv"
    time_series = pd.read_csv(file, index_col='Date')
    ins = Instrument(id, ticker, "Stock", time_series)
    instruments.append(ins)

# Simulation ...
simulator = Simulator(instruments)
simulator.calibrate()
simulator.run_simulation(time_steps=25, number_of_simulations=25)

# Portfolio creation...
holdings = 2, 1, 1, 1, 1, 1, 1, 1
# TODO: Check why 0 holdings does not work well for any model.
holdings = dict(zip(instruments, holdings))
cash = 100
portfolio = Portfolio(holdings, cash)

# Risk assessor creation...
assessor = RiskAssessor(portfolio=portfolio, return_tensor=simulator.return_tensor, time_step=5)
assessor.summary()

# Display reasonableness of simulations...
# for i in range(25):
#     y = simulator.price_tensor[0,:,i]
#     x = [i for i in range(len(y))]
#     pyplot.plot(x, y)
# pyplot.show()

# Mock prices...
price_tensor = torch.tensor([ [[1000]], [[0]], [[0]], [[0]]])
inital_prices = torch.tensor([ [[10]], [[10]], [[10]], [[10]]])

# Create optimizer with different optimization models...
optimizer = SPMaximumUtility(portfolio, simulator.price_tensor, simulator._inital_prices, gamma=1)
# optimizer.solve()

print()
optimizer = SPMaximumUtility(portfolio, simulator.price_tensor, simulator._inital_prices, gamma=-2)
# optimizer.solve()

print()
optimizer = MPCMaximumUtility(portfolio, simulator.return_tensor, gamma=1)
# optimizer.solve()


print("OK!")
