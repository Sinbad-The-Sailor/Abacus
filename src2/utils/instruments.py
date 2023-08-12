# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from utils.price_history import PriceHistory
from models.model import Model


class RiskFactor:
    model: Model

    def __init__(self, identifier: str, price_history: PriceHistory):
        self.identifier = identifier
        self.price_history = price_history

    def __str__(self) -> str:
        return f"{self.identifier}"

    def __repr__(self) -> str:
        return f"{self.identifier}"



class Asset(ABC):
    identifier: str
    risk_factors: list[RiskFactor]

    @property
    @abstractmethod
    def price(self):
        ...

    def __str__(self) -> str:
        return f"Asset: {self.identifier} ({type(self).__name__}) \nRisk Factors: {self.risk_factors}"

    def __repr__(self) -> str:
        return f"{self.identifier}"



class Stock(Asset):
    def __init__(self, identifier, risk_factors):
        self.identifier = identifier
        self.risk_factors = risk_factors

    @property
    def price(self):
        ...
