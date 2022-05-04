"""
TODO: Implement data parser from Refinitiv Eikon to abacus_database.db automatically.
"""
import sqlite3
import pandas as pd

from sqlite3 import Error


def create_connection(db_file: str) -> object:
    try:
        connection = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return connection


def select_all_tasks(connection: object) -> list:
    cur = connection.cursor()
    cur.execute("SELECT * FROM asset_model_student_init_solutions")
    rows = cur.fetchall()
    for row in rows:
        print(row[4])


def select_init_solution(connection: object, asset: str) -> list:
    cur = connection.cursor()
    cur.execute("SELECT * FROM asset_model_student_init_solutions WHERE ASSET = '%s'" % asset)
    row = cur.fetchall()
    return row[0][2:]


def write_final_solution(connection: object, params):
    cur = connection.cursor()
    sql = '''INSERT INTO asset_model_student_solutions(asset, mu, omega, beta0, beta1, beta2, kappa, lambda, nu) 
             VALUES(?, ?, ?, ?, ?, ?, ?, ?, ? )
          '''
    cur.execute(sql, params)
    connection.commit()
    return cur.lastrowid


def select_price_data(connection: object, asset: str) -> list:
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
