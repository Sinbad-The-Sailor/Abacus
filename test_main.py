# -*- coding: utf-8 -*-
import numpy as np
from abacus.simulator.gjr_grach import GJRGARCH
from abacus.simulator.ma import MA
from abacus.simulator.ar import AR
from abacus.simulator.garch import GARCH
from abacus.simulator.model_factory import ModelFactory
from abacus.simulator.nnar import NNAR
from abacus.utilities.dataloader import DataLoader

from matplotlib import pyplot as plt


def test_main():
    start = "2005-12-28"
    end = "2022-07-11"
    interval = "wk"
    instrument_specification = {
        "XOM": {"Currency": "USD", "Type": "Equity"},
        "CVX": {"Currency": "USD", "Type": "Equity"},
        "MSFT": {"Currency": "USD", "Type": "Equity"},
        "WFC": {"Currency": "USD", "Type": "Equity"},
    }
    print("Downloading data...")
    dataloader = DataLoader(start=start, end=end, interval=interval)
    instruments = dataloader.load_yahoo_data(instrument_specification)

    print("Fitting models...")

    # Testing AR/MA model

    instrument = instruments[0]
    model_factory = ModelFactory(instruments)
    # model_factory.build_all()
    ar_model = AR(np.array(instrument.log_return_history), 5)
    ar_model.fit_model()
    print(ar_model.mse)
    print(ar_model.solution)

    # returns = np.array(instrument.log_return_history)
    # garch_model = GJRGARCH(instrument.log_return_history)
    # garch_model.fit_model()

    # print(type(garch_model).__name__)

    # plt.plot(returns)
    # plt.plot(range(len(returns), len(returns) + 2000), garch_model.run_simulation(2000))
    # plt.show()

    # print(garch_model.transform_to_uniform())
    # plt.plot(garch_model.transform_to_true(garch_model.transform_to_uniform()))
    # plt.show()

    # historical_prices = (np.exp(np.cumsum(instrument.log_return_history))*instrument.price_history[0])
    # plt.plot(np.array(historical_prices))
    # plt.plot(range(len(historical_prices), len(historical_prices)+200), np.exp(np.cumsum(garch_model.run_simulation(200)))*historical_prices[-1])
    # plt.show()

    # ma_model = MA(np.array(instrument.log_return_history), 2)
    # ar_model = AR(np.array(instrument.log_return_history), 20)
    # ar_model.fit_model()
    # plt.plot(ar_model.run_simulation(125))
    # plt.show()
    # print(ar_model.mse)
    # historical_prices = (np.exp(np.cumsum(instrument.log_return_history))*instrument.price_history[0])
    # plt.plot(np.array(historical_prices))
    # plt.plot(range(len(historical_prices), len(historical_prices)+200), np.exp(np.cumsum(ar_model.run_simulation(200)))*historical_prices[-1])
    # plt.show()

    # nnar_model = NNAR(returns, 9)
    # nnar_model.fit_model()

    # steps = 25
    # simulated_values = nnar_model.run_simulation(steps)
    # plt.plot(returns)
    # plt.plot(range(len(returns), len(returns) + steps), simulated_values)
    # plt.show()

    # historical_prices = (np.exp(np.cumsum(instrument.log_return_history))*instrument.price_history[0])
    # plt.plot(range(len(historical_prices), len(historical_prices)+steps), np.exp(np.cumsum(simulated_values))*historical_prices[-1])
    # plt.plot(range(len(historical_prices), len(historical_prices)+steps), np.exp(np.cumsum(nnar_model.run_simulation(steps)))*historical_prices[-1])
    # plt.plot(np.array(historical_prices))
    # plt.show()

    # print(nnar_model.mse)
    # print(nnar_model.transform_to_uniform())
    # plt.plot(nnar_model.transform_to_true(nnar_model.transform_to_uniform()))
    # plt.show()


if __name__ == "__main__":
    test_main()
