from matplotlib import pyplot as plt

PLOT_STYLE = ['science', 'notebook', 'grid']


def run_configuration():
    plt.rcParams['text.usetex'] = True
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.style.use(PLOT_STYLE)
