from matplotlib import pyplot as plt
from database import database_parser

PLOT_STYLE = ['science', 'notebook', 'grid']
ABACUS_DATABASE_CONNECTION = database_parser.create_connection('database/abacus_database.db')


def run_configuration():
    plt.rcParams['text.usetex'] = True
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.style.use(PLOT_STYLE)
