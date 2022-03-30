
from data_paser import parse_yahoo_data
from equity_model import StockData, EquityModel


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

    for eq in equity_models:
        eq.fit_model()
        eq.plot_volatility()
