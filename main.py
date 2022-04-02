
from data_paser import parse_yahoo_data
from equity_model import StockData, EquityModel
from tqdm import tqdm
from portfolio import Portfolio

if __name__ == '__main__':
    # Testing sensibility of GJR-GARCH modelling by assets in different asset classes.
    # Note: Testing assets include non-equity class assets.
    assets_ric = ['^GSPC', '^IXIC', '^RUT', '^N225',
                  'GS', 'XOM', 'WFC', 'MS', 'KO', 'T', 'VZ',
                  'AUDUSD=X', 'EURUSD=X', 'JPYUSD=X', 'NOKUSD=X', 'GBPUSD=X',
                  'CL=F', 'GC=F', 'SI=F',
                  '^TNX']
    portfolio = Portfolio(assets_ric=assets_ric)
    portfolio.load_yahoo_data()
    portfolio.fit_asset_models()
    portfolio.plot_volatilities()


    #assets = [StockData(ric) for ric in assets_ric]
    #for asset in tqdm(assets, smoothing=True, colour="green", ncols=50, bar_format=''):
    #    parse_yahoo_data(asset)
    #    # print("Data Fetched for asset: " + str(asset.ric))
    #equity_models = [EquityModel(asset) for asset in assets]

    #for counter, eq_model in enumerate(equity_models):
    #    print(f'Initiation Model {counter + 1}')
    #    if counter != 3 or counter != 9:
    #        eq_model.fit_model(model="normal poisson mixture")
    #        print(f'Model {counter + 1 }: OK!')
    #    else:
    #        print('Model 4: not ok!')
    #for eq_model in equity_models:
     #   eq_model.plot_volatility()
