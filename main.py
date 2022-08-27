# -*- coding: utf-8 -*-
import os
import pymysql
import numpy as np

from abacus.simulator.forecaster import Forecaster
from abacus.optimizer.policies import MPCDummy, MPCLogUtility
from abacus.utilities import email_service
from abacus.utilities.dataloader import DataLoader


def main():
    start = "2005-12-28"
    end = "2022-07-11"
    interval = "wk"
    instrument_specification = {
        "XOM": {"Currency": "USD", "Type": "Equity"},
        "CVX": {"Currency": "USD", "Type": "Equity"},
        "^GSPC": {"Currency": "USD", "Type": "Equity"},
        "WFC": {"Currency": "USD", "Type": "Equity"},
        "MSFT": {"Currency": "USD", "Type": "Equity"},
        "CAT": {"Currency": "USD", "Type": "Equity"},
        "NOC": {"Currency": "USD", "Type": "Equity"},
        "KO": {"Currency": "USD", "Type": "Equity"},
    }

    # Load data from yahoo.
    dataloader = DataLoader(start=start, end=end, interval=interval)
    instruments = dataloader.load_yahoo_data(instrument_specification)

    forc = Forecaster(instruments=instruments, number_of_steps=5)
    forecast = forc.forecast_returns()

    inital_portfolio = np.insert(np.zeros(len(instruments)), 0, 1)
    mpc = MPCDummy(forecast=forecast, inital_portfolio=inital_portfolio)
    mpc.optimize()
    print(mpc.solution)

    print("==========")
    mpc_util = MPCLogUtility(forecast=forecast, inital_portfolio=inital_portfolio)
    mpc_util.optimize()
    print(mpc_util.solution)
    print("==========")

    email_service.send_email(msg=str(mpc_util.solution), status="OK")


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


def aws_database_connection():
    try:
        connection = pymysql.connect(
            host=os.getenv("AWS_DB_HOST"),
            port=int(os.getenv("AWS_DB_PORT")),
            user=os.getenv("AWS_DB_USER"),
            passwd=os.getenv("AWS_DB_PASW"),
            db=os.getenv("AWS_DB_NAME"),
        )
        cursor = connection.cursor()
        sql = "DESCRIBE Assets"
        print(cursor.execute(sql))
        print("[+] RDS Connection Successful")
        connection.close()
    except Exception as e:
        print(f"[-] RDS Connection Failed: {e}")
