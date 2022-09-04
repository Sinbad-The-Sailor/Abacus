# -*- coding: utf-8 -*-
import numpy as np
from abacus.simulator.ma import MA
from abacus.simulator.ar import AR
from abacus.utilities.dataloader import DataLoader

from matplotlib import pyplot as plt


def test_main():
    start = "2005-12-28"
    end = "2022-07-11"
    interval = "wk"
    instrument_specification = {
        "XOM": {"Currency": "USD", "Type": "Equity"},
    }
    print("Downloading data...")
    dataloader = DataLoader(start=start, end=end, interval=interval)
    instruments = dataloader.load_yahoo_data(instrument_specification)

    print("Fitting models...")
    # Testing MA model
    instrument = instruments[0]
    #model = MA(np.array(instrument.log_return_history), 2)
    model = AR(np.array(instrument.log_return_history), 20)
    print(model.fit_model())
    plt.plot(model.run_simulation(125))
    plt.show()
    print(model.transform_to_uniform())
    plt.plot(model.transform_to_true(model.transform_to_uniform()))
    plt.show()

    # plt.plot(model._generate_residuals(model.solution))
    # plt.plot(range(len(instrument.log_return_history), len(instrument.log_return_history)+1500), model.run_simulation(1500))
    # plt.show()

    # historical_prices = (np.exp(np.cumsum(instrument.log_return_history))*instrument.price_history[0])
    # prices = np.exp(np.cumsum(model.run_simulation(152)))*historical_prices[-1]
    # plt.plot(np.array(historical_prices))
    # plt.plot(range(len(historical_prices), len(historical_prices)+152), prices)
    # plt.plot(range(len(historical_prices), len(historical_prices)+152), np.exp(np.cumsum(model.run_simulation(152)))*historical_prices[-1])
    # plt.plot(range(len(historical_prices), len(historical_prices)+152), np.exp(np.cumsum(model.run_simulation(152)))*historical_prices[-1])
    # plt.plot(range(len(historical_prices), len(historical_prices)+152), np.exp(np.cumsum(model.run_simulation(152)))*historical_prices[-1])
    # plt.plot(range(len(historical_prices), len(historical_prices)+152), np.exp(np.cumsum(model.run_simulation(152)))*historical_prices[-1])
    # plt.plot(range(len(historical_prices), len(historical_prices)+152), np.exp(np.cumsum(model.run_simulation(152)))*historical_prices[-1])
    # plt.plot(range(len(historical_prices), len(historical_prices)+152), np.exp(np.cumsum(model.run_simulation(152)))*historical_prices[-1])
    # plt.show()


if __name__ == "__main__":
    test_main()
