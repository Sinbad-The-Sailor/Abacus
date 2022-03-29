import numpy as np

from data_paser import parse_yahoo_data
from equity_model import StockData, EquityModel
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # Testing sensibility of GJR-GARCH modelling by assets in different asset classes.
    # Note: Testing assets include non-equity class assets.
    assets_ric = ['^GSPC', '^IXIC', '^RUT', '^N225',
                  'GS', 'XOM', 'WFC', 'MS', 'KO', 'T', 'VZ',
                  'AUDUSD=X', 'EURUSD=X', 'JPYUSD=X', 'NOKUSD=X', 'GBPUSD=X',
                  'CL=F', 'GC=F', 'SI=F',
                  '^TNX']
    assets = [StockData(ric) for ric in assets_ric]
    for asset in assets:
        parse_yahoo_data(asset)
    equity_models = [EquityModel(asset) for asset in assets]

    # Testing validity of volatility estimations for asset in assets.
    def plot_vol(data, params):
        omg = params[0]
        alp = params[1]
        bet = params[2]
        gam = params[3]

        time = data.index[1:]
        data = data.to_list()
        vol = []
        curr_vol = (omg
                    + alp * (data[0] ** 2)
                    + gam * (data[0] ** 2) * np.where(data[0], 1, 0)
                    + bet * (data[0] ** 2))
        vol.append(curr_vol)

        for i in range(2, len(data)):
            curr_vol = (omg
                        + alp * (data[i-1] ** 2)
                        + gam * (data[i-1] ** 2) * np.where(data[i-1] < 0, 1, 0)
                        + bet * (vol[i-2]))
            vol.append(curr_vol)

        vol = np.sqrt(vol)
        plt.plot(time[20:], vol[20:])
        plt.show()

    for eq in equity_models:
        plt.title(eq.stock_data.ric)
        plot_vol(eq.stock_data.get_log_returns(), eq.fit_model().x)

