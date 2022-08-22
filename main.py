# -*- coding: utf-8 -*-
import os
import pymysql

from abacus.instruments import Equity
from abacus.simulator.forecaster import Forecaster
from abacus.optimizer.policies import MPCDummy, MPCLogUtility
from abacus.utilities import email_service


def main():
    msg = email_service._build_header("OK")
    email_service.send_email(msg=msg)


def main2():
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
    stock6 = Equity(
        ric="MSFT", currency="USD", start_date=start, end_date=end, interval=interval
    )
    stock7 = Equity(
        ric="CAT", currency="USD", start_date=start, end_date=end, interval=interval
    )
    stock8 = Equity(
        ric="NOC", currency="USD", start_date=start, end_date=end, interval=interval
    )
    stock9 = Equity(
        ric="KO", currency="USD", start_date=start, end_date=end, interval=interval
    )
    stock10 = Equity(
        ric="JNJ", currency="USD", start_date=start, end_date=end, interval=interval
    )
    stock11 = Equity(
        ric="AXP", currency="USD", start_date=start, end_date=end, interval=interval
    )
    stock12 = Equity(
        ric="WMT", currency="USD", start_date=start, end_date=end, interval=interval
    )

    instruments = [stock1, stock2, stock3, stock4,
                   stock6, stock7, stock8, stock9, stock10, stock11, stock12]

    forc = Forecaster(instruments=instruments, number_of_steps=5)
    forecast = forc.forecast_returns()

    import numpy as np
    inital_portfolio = np.insert(np.zeros(len(instruments)), 0, 1)
    mpc = MPCDummy(forecast=forecast, inital_portfolio=inital_portfolio)
    mpc.optimize()
    print(mpc.solution)

    print("==========")
    mpc_util = MPCLogUtility(
        forecast=forecast, inital_portfolio=inital_portfolio)
    mpc_util.optimize()
    print(mpc_util.solution)
    print("==========")


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
        connection = pymysql.connect(host=os.getenv("AWS_DB_HOST"),
                                     port=int(os.getenv("AWS_DB_PORT")),
                                     user=os.getenv("AWS_DB_USER"),
                                     passwd=os.getenv("AWS_DB_PASW"),
                                     db=os.getenv("AWS_DB_NAME")
                                     )
        cursor = connection.cursor()
        sql = "DESCRIBE Assets"
        print(cursor.execute(sql))
        print('[+] RDS Connection Successful')
        connection.close()
    except Exception as e:
        print(f'[-] RDS Connection Failed: {e}')
