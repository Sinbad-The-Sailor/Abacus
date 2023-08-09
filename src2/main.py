# -*- coding: utf-8 -*-
import torch

from datetime import datetime
from matplotlib import pyplot as plt

from utils.config import TEST_YAHOO_STOCK_UNIVERSE_16
from utils.stock_factory import StockFactory
from utils.data_handler import YahooDataHandler
from utils.portfolio import Portfolio
from simulator.simulator import Simulator
from optimizer.optimizer import Optimizer
from optimizer.enums import OptimizationModels



# Obtaining test data.
yhdr = YahooDataHandler()
start = datetime.strptime("2013-05-01", r"%Y-%m-%d")
end = datetime.strptime("2023-06-01", r"%Y-%m-%d")
instrument_tickers = TEST_YAHOO_STOCK_UNIVERSE_16
stock_specifications = {ticker: yhdr.get_price_history(ticker, start, end) for ticker in instrument_tickers}

# Build stock instruments.
stock_factory = StockFactory(stock_specifications=stock_specifications)
stocks = stock_factory.build_stocks()

# Build current portfolio.
holdings = {stock: 0 for stock in stocks}
cash = 10_000_000
portfolio = Portfolio(holdings, cash)

# Simulation specifiation.
number_of_simulations = 100
time_steps = 14

# Simulator for returns and prices.
simulator = Simulator(instruments=stocks)
simulator.calibrate()
simulator.run_simulation(time_steps=time_steps, number_of_simulations=number_of_simulations)
return_tensor = simulator.return_tensor
price_tensor = simulator.price_tensor

# Plotting of price simulations.
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

# Selecting optimzation model.
optimizer = Optimizer(portfolio=portfolio, simulation_tensor=price_tensor)
optimizer.model = OptimizationModels.SP_MAXIMIZE_UTILITY
optimizer.run()
