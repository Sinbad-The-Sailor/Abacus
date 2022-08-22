# -*- coding: utf-8 -*-
import numpy as np

from abacus.simulator import simulator


class Forecaster:

    def __init__(self, instruments, number_of_steps):
        self.instruments = instruments
        self.number_of_steps = number_of_steps
        self.simulator = simulator.Simulator(instruments=instruments)
        self.init_prices = self.simulator._last_prices()

    def forecast_returns(self):
        dim = (len(self.instruments), self.number_of_steps)
        dependency = True
        number_of_simulations = 1000
        result = np.zeros(dim)

        for _ in range(number_of_simulations):
            simulated_matrix = self.simulator.run_simultion_assets(number_of_steps=self.number_of_steps,
                                                                   dependency=True)
            simulated_matrix = np.vstack(simulated_matrix)
            result += simulated_matrix

        result = 1/number_of_simulations * result

        # for _ in range(len(self.instruments)):
        #     prices_XOM = []
        #     prices_XOM.append(self.init_prices[_])
        #     returns_XOM = result[_, :]
        #     for i in range(self.number_of_steps):
        #         prev_price = prices_XOM[i]
        #         return_ = np.prod(np.exp(returns_XOM[:i]))
        #         prices_XOM.append(prev_price*return_)
        #     print(prices_XOM)

        return result

    def forecast_prices(self):
        raise NotImplemented
