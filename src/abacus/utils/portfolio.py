# -*- coding: utf-8 -*-



class Portfolio:

    def __init__(self, holdings: dict[str:int], cash: float):
        self._holdings = holdings
        self._cash = cash

    @property
    def weights(self):
        total_holdings = sum(map(abs, self._holdings.values()))
        if total_holdings == 0:
            return {ticker: 0 for ticker in self._holdings.items()}
        return {ticker: holding/total_holdings for ticker, holding in self._holdings.items()}

    @property
    def indices(self):
        return [instrument.id for instrument in self._holdings]

    @property
    def holdings(self):
        return self._holdings

    @property
    def instruments(self):
        return self._holdings.keys()

    def __str__(self):
        output = "\nPortfolio Holdings\n"
        output += f"Cash: {self._cash}\n\n"
        for ticker, holding in self._holdings.items():
            output += f"{ticker}: {holding}\n\n"
        return output
