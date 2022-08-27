# -*- coding: utf-8 -*-
import os
import pymysql

from abacus.instruments import Instrument


class DataLoader:

    def __init__(self):
        self._has_connection = False

    def load_yahoo_data(instrument_specification: dict) -> list[Instrument]:
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
            pass

        return instruments

    # Using AWS / DB connection with appropriate schema.
    def create_connection(self) -> pymysql.Connection:
        """
        Creates connection to database using environment variables.
        .env-mock gives direction of configuration.

        Returns:
            pymysql.Connection:
        """
        pass

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
        if self._has_connection:
            return True
        return False
