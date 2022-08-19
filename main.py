# -*- coding: utf-8 -*-
from ast import Eq
import numpy as np

from matplotlib import pyplot as plt

from models.equity_models import GARCHEquityModel, GJRGARCHEquityModel
from models.fx_models import GARCHFXModel
from instruments.instruments import Equity, FX, Instrument
from portfolio import Portfolio

from abacus_utils.risk_tools import risk_assessor
from abacus_simulator import simulator


def equity_model_factory(equity: Equity):
    # TODO: More logic based on AIC or BIC e.g.
    initial_parametes_gjr = np.array([0.05, 0.80, 0.001])
    initial_parametes_gar = np.array([0.05, 0.80])
    model = GJRGARCHEquityModel(
        initial_parametes_gjr, equity.log_return_history)
    equity.set_model(model=model)


def main():
    # Create stocks
    start = "2005-12-28"
    end = "2022-07-11"
    interval = "wk"
    stock1 = Equity(
        ric="XOM", currency="USD", start_date=start, end_date=end, interval=interval
    )
    stock2 = Equity(
        ric="CVX", currency="USD", start_date=start, end_date=end, interval=interval
    )
    stock3 = Equity(
        ric="^GSPC", currency="USD", start_date=start, end_date=end, interval=interval
    )
    stock4 = Equity(
        ric="WFC", currency="USD", start_date=start, end_date=end, interval=interval
    )

    instruments = [stock1, stock2, stock3, stock4]

    # SHOULD BE IN SIMULATOR:
    for instrument in instruments:
        equity_model_factory(instrument)

    simulator = Portfolio(instruments=instruments)
    simulator.fit_models()
    simulator.fit_portfolio()

    for instrument in simulator.instruments:
        instrument.model.plot_volatility()

    simulated_portfolio_returns = simulator.run_simulation_portfolio(
        52, 10000, dependency=True
    )
    plt.hist(simulated_portfolio_returns, bins=50)
    plt.show()


if __name__ == "__main__":
    main()


def test_main():
    """Template for finial main/run method."""
    # 1. Initialize config file/run configuration files/pingers.
    running = True

    # 2. Establish connection to db.

    # 3. Establish connection to markets.

    # x. Establish connection to AWS.

    # 4. Load previous portfolio from db.

    while running:

        # 5. Update market information.
        # 6. Commit to db.

        # +++ RUN SIMULATION ENGINE +++
        # 7. Refit all models.
        # 7. Recalibrate all valuation drivers.
        # 9. Produce simulations
        # 8. Commit to db.

        # +++ RUN VALUATION ENGINE +++
        # 9. Update all valuation
        # 10. Commit to db.

        # +++ RUN OPTIMZATION EGINE +++
        # 11. Update portfolio.
        # 12. Commit to db.

        # 13. Update Dashboard.
        # 14. Send report email from logging and resulting transactions.

        # ... wait until next period (1 day, 2 day, 1 week)
        # if any errors occur, send email with logs and stop the process.
        pass
