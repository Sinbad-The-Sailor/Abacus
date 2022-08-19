# -*- coding: utf-8 -*-
from abc import ABC
from dataclasses import dataclass


class Instrument(ABC):

    def __init__(self, ticker, name, ric, ycode, type):
        self.ticker = ticker
        self.name = name
        self.ric = ric
        self.ycode = ycode
        self.type = type
