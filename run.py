# -*- coding: utf-8 -*-
import pandas as pd

from matplotlib import pyplot
from tests.test_config import TEST_YAHOO_STOCK_UNIVERSE_16, TEST_YAHOO_STOCK_UNIVERSE_8, TEST_YAHOO_STOCK_UNIVERSE_4
from src.abacus.utils.instrument import Instrument
from src.abacus.utils.portfolio import Portfolio
from src.abacus.simulator.simulator import Simulator
from src.abacus.assessor.risk_assessor import RiskAssessor
from src.abacus.optimizer.optimizer import SPMaximumUtility



# Mock instrument creation...
instruments = []
for id, ticker in enumerate(sorted(TEST_YAHOO_STOCK_UNIVERSE_4)):
    file = f"tests/data/{ticker}.csv"
    time_series = pd.read_csv(file, index_col='Date')
    ins = Instrument(id, ticker, "Stock", time_series)
    instruments.append(ins)

# Simulation ...
simulator = Simulator(instruments)
simulator.calibrate()
simulator.run_simulation(time_steps=10, number_of_simulations=1000)

# Portfolio creation...
holdings = 500, 200, 100, 125
holdings = dict(zip(instruments[0:4], holdings))
print(holdings)
cash = 100_000
portfolio = Portfolio(holdings, cash)

# Risk assessor creation...
assessor = RiskAssessor(portfolio=portfolio, return_tensor=simulator.return_tensor, time_step=5)
assessor.summary()

# Display reasonableness of simulations...
for i in range(25):
    y = simulator.price_tensor[0,:,i]
    x = [i for i in range(len(y))]
    pyplot.plot(x, y)
pyplot.show()

# Create optimizer with different optimization models...
optimizer = SPMaximumUtility(portfolio, simulator.price_tensor)
optimizer.solve()



print("OK!")
