import numpy as np

from data_paser import parse_yahoo_data
from equity_model import StockData, EquityModel
from matplotlib import pyplot as plt


if __name__ == '__main__':
    assets_ric = ['GS', 'XOM', '^GSPC', 'AUDUSD=X', 'EURUSD=X', 'WFC']
    assets = [StockData(ric) for ric in assets_ric]
    for asset in assets:
        parse_yahoo_data(asset)
    equity_models = [EquityModel(asset) for asset in assets]

    # Testing validity of volatility estimations for asset in assets.
    def plot_vol(data, params):
        omg = params[0]
        alp = params[1]
        bet = params[2]

        time = data.index[1:]
        data = data.to_list()
        vol = []
        curr_vol = omg + alp * (data[0] ** 2) + bet * (data[0] ** 2)
        vol.append(curr_vol)

        for i in range(2, len(data)):
            curr_vol = omg + alp * (data[i-1] ** 2) + bet * (vol[i-2])
            vol.append(curr_vol)

        vol = np.sqrt(vol)
        plt.plot(time[15:], vol[15:])
        plt.show()

    for eq in equity_models:
        plot_vol(eq.stock_data.get_log_returns(), eq.fit_model().x)
