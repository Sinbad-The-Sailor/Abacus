# -*- coding: utf-8 -*-
import numpy as np

from pandas import DataFrame
from datetime import date
from src.abacus.utils.instrument import Instrument



class Universe:

    def __init__(self, instrument_specifications: dict[str:DataFrame], date_today: str=str(date.today())):
        self._instrument_specifications = instrument_specifications
        self._date_today = date_today
        self._instruments = None
        self._instrument_build_date = None

    @property
    def date_today(self):
        return self._date_today

    @date_today.setter
    def date_today(self, new):
        self._date_today = new

    @property
    def instrument_identifiers(self):
        return sorted(self._instrument_specifications.keys())

    @property
    def instruments(self) -> list[Instrument]:
        if self.has_updated_cache():
            return self._instruments

        built_instruments = []
        for id, identifier in enumerate(self.instrument_identifiers):
            time_series = self._instrument_specifications[identifier].loc[:str(self.date_today)]
            ins = Instrument(id, identifier, "Stock", time_series)
            built_instruments.append(ins)
        return built_instruments

    @property
    def todays_returns(self):
        return np.array([instrument.art_returns[-1] for instrument in self.instruments]).flatten()

    def has_updated_cache(self) -> bool:
        has_cache = self._instruments is not None
        has_last_build = self._instrument_build_date is not None
        updated_date = self._date_today == self._instrument_build_date
        conditions = has_cache, has_last_build, updated_date
        return any(conditions)
