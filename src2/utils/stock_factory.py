# -*- coding: utf-8 -*-
from utils.instruments import Stock, RiskFactor


class StockFactory:

    def __init__(self, stock_specifications):
        self._stock_specifications = stock_specifications

    def build_stocks(self) -> list[Stock]:
        stocks = []
        id = 0
        for ticker, data in self._stock_specifications.items():
            identifier = f"{ticker}_RF"
            risk_factor = RiskFactor(identifier, data)
            stock = Stock(id=id, identifier=ticker, risk_factors=[risk_factor])
            id += 1
            stocks.append(stock)
        return stocks
