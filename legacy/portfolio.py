# -*- coding: utf-8 -*-
from legacy.equity_model import EquityModel
from legacy.stock_data import StockData
from tqdm import tqdm


class Portfolio:
    def __init__(self, assets_ric=[], assets=[], assets_models=[]):
        if assets_ric:
            self.assets_ric = assets_ric
        self.assets = assets
        self.asset_models = assets_models

    def load_asset_data(self):
        for ric in tqdm(self.assets_ric, desc="Creating Data Objects.".ljust(25), colour='CYAN'):
            self.assets.append(StockData(ric))
            break

        for asset in tqdm(self.assets, desc="Creating Asset Models.".ljust(25), colour='CYAN'):
            self.asset_models.append(EquityModel(asset))
            break

    def fit_asset_models(self):
        for eq_model in tqdm(self.asset_models, desc="MLE fitting models.".ljust(25), colour='CYAN'):
            eq_model.fit_model("student poisson mixture")
            break

    def plot_volatilities(self):
        for eq_model in self.asset_models:
            eq_model.plot_volatility()
            break

    def plot_qqs(self):
        for eq_model in self.asset_models:
            eq_model.plot_qq()
            break
