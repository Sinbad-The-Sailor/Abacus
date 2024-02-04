# -*- coding: utf-8 -*-
import pandas as pd

from matplotlib import pyplot
from tests.test_config import TEST_YAHOO_STOCK_UNIVERSE_16
from src.abacus.utils.instrument import Instrument
from src.abacus.simulator.simulator import Simulator

instruments = []
for ticker in TEST_YAHOO_STOCK_UNIVERSE_16[0:3]:
    file = f"tests/data/{ticker}.csv"
    time_series = pd.read_csv(file, index_col='Date')
    ins = Instrument(ticker, "Stock", time_series)
    instruments.append(ins)

simulator = Simulator(instruments)
simulator.calibrate()
simulator.run_simulation(500, 50)


for i in range(50):
    y = simulator.price_tensor[0,:,i]
    x = [i for i in range(len(y))]
    pyplot.plot(x, y)
pyplot.show()


print("OK!")
