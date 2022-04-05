
from portfolio import Portfolio

if __name__ == '__main__':
    print('running')
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
