# A portfolio should be created in main and fed a list of RICs to fetch data and "have" 1..n assets. This portfolio
# should only have few functions such as, VAR monte carlo EVT, which starts running the VaR calculations with simulation
# techniques.

class Portfolio:
    assets = []
    weights = []
    portfolio_value = []

    def __int__(self, asset_ric=[], weights=[], portfolio_value=0):
        self.portfolio_value = portfolio_value
        self.weights = weights
