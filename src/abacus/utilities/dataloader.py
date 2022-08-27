# -*- coding: utf-8 -*-
import os
import pymysql

from pandas_datareader import data as pdr
from abacus.instruments import Instrument, Equity, FX
from abacus.utilities.instrument_enum import InstrumentType
from abacus.utilities.currency_enum import Currency

import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loading class to fetch and market and portfolio data from different providers.
    """

    def __init__(self, start, end, interval):
        self.has_connection = False
        self.start = start
        self.end = end
        self.interval = interval

    def load_yahoo_data(self, instrument_specification: dict) -> list[Instrument]:
        """
        Returns a list of instruments with historical price data. Data fetched
        from Yahoo Finance.

        Args:
            instrument_specification (dict): specifying code, type and currency for instruments.

        Returns:
            list[Instrument]: A list of insturments with historical data.
        """
        instruments = []
        for key, value in instrument_specification.items():
            current_code = key
            current_currency = value["Currency"]
            current_type = value["Type"]
            current_instrument = self._yahoo_instrument_builder(
                type=current_type,
                code=current_code,
                currency=current_currency,
                start=self.start,
                end=self.end,
                interval=self.interval,
            )

            try:
                price_history = pdr.get_data_yahoo(
                    current_code, start=self.start, end=self.end, interval=self.interval
                )["Adj Close"]
                current_instrument.set_price_history(price_history=price_history)
            except:
                logger.error("Cannot fetch yahoo data.")

            instruments.append(current_instrument)

        return instruments

    def _yahoo_instrument_builder(
        self, type: str, code: str, currency: str, start: str, end: str, interval: str
    ) -> Instrument:
        """
        Creates a instrument instance depending on type specified.

        Args:
            type (str): Type of instrument.
            code (str): Code for yfiance identification.
            currency (str): Local currency of instrument.
            start (str): Start date.
            end (str): End date.
            interval (str): Time interval.

        Raises:
            ValueError: If instrument type is unrecognized.

        Returns:
            Instrument: Built instrument.
        """

        if InstrumentType[type].name == "Equity":
            return Equity(code, Currency[currency].value, start, end, interval)
        elif InstrumentType[type].name == "FX":
            return FX(code, Currency[currency].value, start, end, interval)
        else:
            raise ValueError("Instrument Type not recognized.")

    # Using AWS / DB connection with appropriate schema.
    def create_connection(self) -> pymysql.Connection:
        """
        Creates connection to database using environment variables.
        .env-mock gives direction of configuration.

        Returns:
            pymysql.Connection:
        """
        try:
            connection = pymysql.connect(
                host=os.getenv("AWS_DB_HOST"),
                port=int(os.getenv("AWS_DB_PORT")),
                user=os.getenv("AWS_DB_USER"),
                passwd=os.getenv("AWS_DB_PASW"),
                db=os.getenv("AWS_DB_NAME"),
            )
            cursor = connection.cursor()
            sql = "DESCRIBE Assets"
            print(cursor.execute(sql))
            print("[+] RDS Connection Successful")
            connection.close()
        except Exception as e:
            print(f"[-] RDS Connection Failed: {e}")

        return None

    def load_instrument_codes():
        pass

    def update_instrument_data():
        pass

    def insert_instrument_data():
        pass

    def load_instrument_data():
        pass

    def insert_portfolio():
        pass

    def load_portfolio():
        pass

    def load_current_portfolio():
        pass

    def _has_connection(self) -> bool:
        """
        Checks if connection to db is established.

        Returns:
            bool: connection status.
        """
        if self.has_connection:
            return True
        return False
