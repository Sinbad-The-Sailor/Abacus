# -*- coding: utf-8 -*-
from ast import Eq
import numpy as np

from matplotlib import pyplot as plt

from models.equity_models import GARCHEquityModel, GJRGARCHEquityModel
from models.fx_models import GARCHFXModel
from instruments.instruments import Equity, FX
from portfolio import Portfolio
from risk_utils.risk_assessor import RiskAssessor


def main():

    # CREATE ASSETS.
    start = "2005-12-28"
    end = "2022-07-11"
    interval = "wk"

    stock1 = Equity(ric="XOM", currency="USD", start_date=start,
                    end_date=end, interval=interval)
    stock2 = Equity(ric="CVX", currency="USD", start_date=start,
                    end_date=end, interval=interval)
    stock3 = Equity(ric="^GSPC", currency="USD",
                    start_date=start, end_date=end, interval=interval)

    fx1 = FX(ric="USDEUR=X", currency="USD", start_date=start,
             end_date=end, interval=interval)
    fx2 = FX(ric="USDGBP=X", currency="USD", start_date=start,
             end_date=end, interval=interval)

    stock4 = Equity(ric="XOM", currency="USD", start_date=start,
                    end_date=end, interval=interval)
    initial_parametes_gjr = np.array(
        [0.05, 0.80, 0.001])

    # CREATE MODELS FOR EACH ASSET.
    initial_parametes = np.array(
        [0.05, 0.80])

    model_XOM = GARCHEquityModel(
        initial_parameters=initial_parametes, data=stock1.log_return_history)
    model_BOA = GARCHEquityModel(
        initial_parameters=initial_parametes, data=stock2.log_return_history)
    model_SPX = GARCHEquityModel(
        initial_parameters=initial_parametes, data=stock3.log_return_history)
    model_XOM_New = GJRGARCHEquityModel(
        initial_parameters=initial_parametes_gjr, data=stock4.log_return_history
    )

    model_EUR = GARCHFXModel(
        initial_parameters=initial_parametes, data=fx1.log_return_history)
    model_GBP = GARCHFXModel(
        initial_parameters=initial_parametes, data=fx2.log_return_history)

    # SET MODEL FOR EACH ASSET.
    stock1.set_model(model_XOM)
    stock2.set_model(model_BOA)
    stock3.set_model(model_SPX)
    stock4.set_model(model_XOM_New)

    fx1.set_model(model_EUR)
    fx2.set_model(model_GBP)

    # CREATE PORTFOLIO AND RUN.
    instruments = [stock1, stock2, stock3, stock4]
    holdings = np.array([10, 10, 100, 10])

    portfolio = Portfolio(instruments=instruments, holdings=holdings)

    portfolio.fit_models()
    portfolio.fit_portfolio()

    for instrument in portfolio.instruments:
        instrument.model.plot_volatility()

    simulated_portfolio_returns = portfolio.run_simulation_portfolio(
        1, 10000, dependency=True)

    plt.hist(simulated_portfolio_returns, bins=50)
    plt.show()

    risk_calc = RiskAssessor(simulated_portfolio_returns)
    risk_calc.risk_summary()


if __name__ == "__main__":
    main()


def test_main():
    """Template for finial main/run method."""
    # 1. Initialize config file.
    running = True

    # 2. Establish connection to db.

    # 3. Establish connection to markets.

    # 4. Load previous portfolio.

    while running:

        # 5. Update market information.

        # 6. Commit to db.

        # 7. Refit all models.

        # 8. Commit to db.

        # 7. Recalibrate all valuation drivers.

        # 8. Commit to db.

        # 9. Update all valuation

        # 10. Commit to db.

        # 11. Update portfolio.

        # 12. Commit to db.

        # 13. Update Dashboard.

        # 14. Send report email.

        # ... wait until next period (1 day, 2 day, 1 week)
        # if any errors occur, send email with logs and stop the process.
        pass
