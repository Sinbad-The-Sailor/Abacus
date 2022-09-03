# -*- coding: utf-8 -*-
import numpy as np
from abacus.simulator.new_model_selection import MA
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
    model = MA(np.array(instrument.log_return_history), 2)
    print(model.fit_model())

    plt.plot(instrument.log_return_history)
    plt.show()







































if __name__ == "__main__":
    test_main()
