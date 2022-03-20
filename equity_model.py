# An equity model can be thought of BEING a model of a STOCK itself.
# The stock will have some price history, ticker, bid-ask spreads, etc, currency, ISIN
# This data will primarily exist in the StockData class, which will be fed into the StockModel for further usage

# When creating an equity model you will feed it data from a stock_data object.


class StockData:
    # List of data in ascending order! Since dicts are hash-tables one might think about using dicts with
    # dates as a key
    time_frame = []
    bid_price, ask_price, mid_price = [], [], []
    arithmetic_returns = []
    log_returns = []
    ticker = ''
    isin = ''

    def __init__(self):
        pass


class EquityModel:

    def __init__(self):
        pass

    def run_simulation(self):
        pass

    def fit_model(self):
        pass
