# -*- coding: utf-8 -*-
import torch

from datetime import datetime
from matplotlib import pyplot as plt

from utils.config import YAHOO_STOCK_UNIVERSE_16
from utils.stock_factory import StockFactory
from simulator.simulator import Simulator
from optimizer.optimizer import Optimizer
from optimizer.enums import OptimizationModels



start = datetime.strptime("2013-05-01", r"%Y-%m-%d")
end = datetime.strptime("2023-06-01", r"%Y-%m-%d")
instrument_specification = YAHOO_STOCK_UNIVERSE_16
instrument_factory = StockFactory(tickers=instrument_specification, start=start, end=end)
stocks = instrument_factory.build_stocks()

number_of_simulations = 1000
time_steps = 14
sim = Simulator(stocks)
sim.calibrate()
sim.run_simulation(time_steps, number_of_simulations)
return_tensor = sim.return_tensor
price_tensor = sim.price_tensor

fig, ax = plt.subplots(4, 4)
ix, iy = 0, 0
for i, stock in enumerate(stocks):
    stock_name = stock.identifier
    past_prices = stock.risk_factors[0].price_history.mid_history[-500:]
    past_time = range(len(past_prices))
    ax[ix, iy].plot(past_time, past_prices)
    ax[ix, iy].set_title(stock_name)

    future_time = range(len(past_time) - 1, len(past_time) + time_steps)
    inital_price = torch.tensor(past_prices[-1])
    for k in range(number_of_simulations):
        all_prices = torch.empty(time_steps+1)
        prices = torch.exp(torch.cumsum(return_tensor[i,:,k], dim=0)) * inital_price
        all_prices[0] = inital_price
        all_prices[1:] = prices
        ax[ix, iy].plot(future_time, all_prices, color="grey")

    if ix != 3:
        ix += 1
    else:
        iy += 1
        ix = 0
plt.tight_layout()
plt.show()

optimizer = Optimizer(optimization_model=OptimizationModels.SP_MAXIMIZE_UTILITY, simulation_tensor=price_tensor)
optimizer.run()
