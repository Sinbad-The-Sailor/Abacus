"""
TODO: Implement data parser from Refinitiv Eikon to abacus_database.db automatically.
"""
import sqlite3
import pandas as pd

from sqlite3 import Error


def create_connection(db_file: str) -> object:
    """
    Establishes connection to database.

    Args:
        db_file: file path to database.

    Returns: connection to database.
    """
    try:
        connection = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return connection


def select_all_assets(connection: object) -> list:
    """
    Selects all assets available in the database.

    Args:
        connection: a connection to the database.

    Returns: list of all assets available in the database.
    """
    cur = connection.cursor()
    cur.execute("SELECT TICKER FROM assets")
    return cur.fetchall()


def select_init_solution(connection: object, asset: str) -> list:
    """
    Selects an initial solution for a Student's t Poisson mixture model. Note, this model is currently used for all
    ETF:s in the database, regardless of the type of asset.

    Args:
        connection: a connection to the database.
        asset: a ric/ticker identifying the asset.

    Returns: list of initial parameters for the model.
    """
    cur = connection.cursor()
    cur.execute("SELECT * FROM asset_model_student_init_solutions WHERE ASSET = '%s'" % asset)
    row = cur.fetchall()
    return row[0][2:]


def write_final_solution(connection: object, params, asset):
    """
    Writes a found solution of model parameters for a Student's t Poisson mixture model to the database. Note, this
    model is currently used for all ETF:s in the database, regardless of the type of asset.

    Args:
        connection: a connection to the database.
        params: a list of parameters for the model.
        asset: a ric/ticker identifying the asset.

    Returns: inserted row id.

    """
    cur = connection.cursor()
    sql = '''UPDATE asset_model_student_solutions SET
             omega = ?,
             beta0 = ?,
             beta1 = ?,
             beta2 = ?,
             mu = ?,
             kappa = ?,
             lamb = ?,
             nu = ?
             WHERE asset = '%s'
          ''' % asset
    cur.execute(sql, params)
    connection.commit()
    return cur.lastrowid


def select_price_data(connection: object, asset: str) -> pd.DataFrame:
    """
    Selects price data from the database for a given asset.

    Args:
        connection: a connection to the database.
        asset: a ric/ticker identifying the asset.

    Returns: a dataframe of prices with corresponding dates.
    """
    cur = connection.cursor()
    cur.execute("SELECT date, close FROM asset_prices WHERE ASSET = '%s'" % asset)
    close_data = [price_tuple[:] for price_tuple in cur.fetchall()]

    # Correcting the order for StockData log-return calculation.
    close_data.reverse()

    # Formatting data as a proper Pandas dataframe.
    df = pd.DataFrame(close_data, columns=['date', 'close'])
    df.set_index('date', inplace=True, drop=True)
    df.index = pd.to_datetime(df.index)
    return df
