# A portfolio should be created in main and fed a list of RICs to fetch data and "have" 1..n assets. This portfolio
# should only have few functions such as, VAR monte carlo EVT, which starts running the VaR calculations with simulation
# techniques.
from data_paser import parse_yahoo_data
from equity_model import StockData, EquityModel
from tqdm import tqdm


class Portfolio:
    assets_ric = []
    assets = []
    asset_models = []
    # weights = []
    # portfolio_value = []

    def __init__(self, assets_ric=[], weights=[], portfolio_value=0):
        if assets_ric:
            self.assets_ric = assets_ric

    def set_assets_ric(self, assets_ric: list):
        self.assets_ric = assets_ric

    def load_yahoo_data(self):
        for ric in tqdm(self.assets_ric, desc="Creating Data Objects.".ljust(25)):
            self.assets.append(StockData(ric))

        for asset in tqdm(self.assets, desc="Downloading Data.".ljust(25)):
            parse_yahoo_data(asset)

        for asset in tqdm(self.assets, desc="Creating Asset Models.".ljust(25), text_pane="cyan"):
            self.asset_models.append(EquityModel(asset))

    def fit_asset_models(self):
        for eq_model in tqdm(self.asset_models, desc="MLE fitting models.".ljust(25)):
            eq_model.fit_model("normal poisson mixture")

    def plot_volatilities(self):
        for eq_model in self.asset_models:
            eq_model.plot_volatility()
