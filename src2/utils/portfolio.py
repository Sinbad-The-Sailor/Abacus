# -*- coding: utf-8 -*-

class Portfolio:

    def __init__(self, holdings: dict[str:int], cash: float):
        self._holdings = holdings
        self._cash = cash
