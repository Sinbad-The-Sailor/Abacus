"""
TODO: Create makefile for initial configuration.
"""
from matplotlib import pyplot as plt
from database.database_parser import create_connection

PLOT_STYLE = ['science', 'notebook', 'grid']
ABACUS_DATABASE_CONNECTION = create_connection('database/abacus_database.db')


def run_configuration():
    plt.rcParams['text.usetex'] = True
    plt.rcParams["figure.figsize"] = (18, 15)
    plt.style.use(PLOT_STYLE)

