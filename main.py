import config as cfg

from portfolio import Portfolio

if __name__ == '__main__':
    cfg.run_configuration()
    # Testing assets.
    assets_ric = ['^N225', '^IXIC', '^RUT', '^GSPC',
                  'GS', 'XOM', 'WFC', 'MS', 'KO', 'T', 'VZ',
                  'AUDUSD=X', 'EURUSD=X', 'JPYUSD=X', 'NOKUSD=X', 'GBPUSD=X',
                  'CL=F', 'GC=F', 'SI=F',
                  '^TNX']

    portfolio = Portfolio(assets_ric=assets_ric)
    portfolio.load_yahoo_data()
    portfolio.fit_asset_models()
    print('Running QQ.')
    portfolio.plot_qqs()
    portfolio.plot_volatilities()
