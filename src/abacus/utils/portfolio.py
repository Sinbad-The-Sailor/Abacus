# -*- coding: utf-8 -*-
from typing import NoReturn



class Portfolio:

    def __init__(self, holdings: dict[str:int]=None, weights: dict[str:float]=None, cash: float=None):
        self._holdings = holdings
        self._weights = weights
        self._cash = cash

    @property
    def weights(self) -> dict[str:float] | NoReturn:
        if not self._weights:
            raise ValueError("No weights are available.")
        return self._weights

    @weights.setter
    def weights(self, new):
        self._weights = new

    @property
    def holdings(self) -> dict[str:int] | NoReturn:
        if not self._weights:
            raise ValueError("No holdings are available.")
        return self._holdings

    @holdings.setter
    def holdings(self, new):
        self._holdings = new

    @property
    def cash(self) -> float | NoReturn:
        if not self._cash:
            raise ValueError("No cash is available.")
        return self._cash

    @cash.setter
    def cash(self, new):
        self._cash = new

    @property
    def weights_from_holdings(self) -> dict[str:float]:
        total_holdings = sum(map(abs, self._holdings.values()))
        if total_holdings == 0:
            return {ticker: 0 for ticker in self._holdings.items()}
        return {ticker: holding/total_holdings for ticker, holding in self._holdings.items()}

    @property
    def indices(self) -> list[int] | NoReturn:
        if self._holdings:
            return [instrument.id for instrument in self._holdings]
        elif self._weights:
            return [instrument.id for instrument in self._weights]
        raise ValueError("Portfolio has no instruments.")

    @property
    def instruments(self) -> list[str] | NoReturn:
        if self._holdings:
            return self._holdings.keys()
        elif self._weights:
            return self._weights.keys()
        raise ValueError("Portfolio has no instruments.")

    def __str__(self):
        output = "\nPortfolio Holdings\n"
        output += f"Cash: {self._cash}\n\n"
        for ticker, holding in self._holdings.items():
            output += f"{ticker}: {holding}\n\n"
        return output
