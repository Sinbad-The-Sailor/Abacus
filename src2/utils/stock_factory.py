# -*- coding: utf-8 -*-

from utils.instruments import Asset, Stock, RiskFactor
from utils.data_handler import YahooDataHandler

class StockFactory:

    def __init__(self, tickers, start, end, interval=None):
        self._tickers = tickers
        self._start = start
        self._end = end
        self._ydr = YahooDataHandler()


    def build_stocks(self) -> list[Stock]:
        stocks = []
        for ticker in self._tickers:
            price_history = self._ydr.get_price_history(ticker, self._start, self._end)
            identifier = f"{ticker}_RF"
            risk_factor = RiskFactor(identifier, price_history)
            stock = Stock(identifier=ticker, risk_factors=[risk_factor])
            stocks.append(stock)
        return stocks
