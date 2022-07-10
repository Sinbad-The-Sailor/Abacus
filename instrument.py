from abc import ABC, abstractmethod
from datetime import datetime

from enums.currency import Currency


class Instrument(ABC):
    isin: str
    start_date: datetime
    end_date: datetime
    local_currency: Currency
