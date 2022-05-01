"""
TODO: Implement data parser from Refinitiv Eikon to abacus_database.db automatically.
"""
import sqlite3
from sqlite3 import Error


def create_connection(db_file: str) -> object:
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        connection = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return connection


def select_all_tasks(connection: object) -> list:
    """
    Query all rows in the tasks table
    :param connection: the Connection object
    :return:
    """
    cur = connection.cursor()
    cur.execute("SELECT * FROM asset_model_student_init_solutions")

    rows = cur.fetchall()

    for row in rows:
        print(row[4])


conn = create_connection('abacus_database.db')
select_all_tasks(conn)
