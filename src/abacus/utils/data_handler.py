# -*- coding: utf-8 -*-
import yfinance as yf

from datetime import datetime
from abc import ABC, abstractmethod
from pandas_datareader import data as pdr

from utils.price_history import PriceHistory



class DataHandler(ABC):
    @staticmethod
    @abstractmethod
    def get_price_history(identifier: str, start_date: datetime, end_date: datetime) -> PriceHistory:
        ...

class YahooDataHandler(DataHandler):
    def __init__(self):
        yf.pdr_override()

    @staticmethod
    def get_price_history(identifier: str, start_date: datetime, end_date: datetime) -> PriceHistory:
        stock_data = pdr.get_data_yahoo(identifier, start=start_date, end=end_date)["Adj Close"]
        return PriceHistory(mid_history=stock_data)

class RefinitivDataHandler(DataHandler):
    def __init__(self):
        ...

    @staticmethod
    def get_price_history(identifier: str, start_date: datetime, end_date: datetime) -> PriceHistory:
        ...

class BloombergDataHandler(DataHandler):
    def __init__(self):
        ...

    @staticmethod
    def get_price_history(identifier: str, start_date: datetime, end_date: datetime) -> PriceHistory:
        ...
