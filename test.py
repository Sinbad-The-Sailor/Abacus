import numpy as np

from test_models import GARCHEquityModel, GARCHFXModel
from test_instruments import Equity, FX
from test_portfolio import Portfolio

from matplotlib import pyplot as plt

from test_risk_assessor import RiskAssessor


def main():

    # CREATE ASSETS.
    start = "2011-12-28"
    end = "2022-07-11"
    interval = "wk"

    stock1 = Equity(ric="XOM", currency="USD", start_date=start,
                    end_date=end, interval=interval)
    stock2 = Equity(ric="GS", currency="USD", start_date=start,
                    end_date=end, interval=interval)
    stock3 = Equity(ric="LQD", currency="USD",
                    start_date=start, end_date=end, interval=interval)

    fx1 = FX(ric="USDEUR=X", currency="USD", start_date=start,
             end_date=end, interval=interval)
    fx2 = FX(ric="USDGBP=X", currency="USD", start_date=start,
             end_date=end, interval=interval)

    # CREATE MODELS FOR EACH ASSET.
    initial_parametes = [0.01, 0.01, 0.7]

    model_XOM = GARCHEquityModel(
        initial_parameters=initial_parametes, data=stock1.log_return_history)
    model_BOA = GARCHEquityModel(
        initial_parameters=initial_parametes, data=stock2.log_return_history)
    model_SPX = GARCHEquityModel(
        initial_parameters=initial_parametes, data=stock3.log_return_history)

    model_EUR = GARCHFXModel(
        initial_parameters=initial_parametes, data=fx1.log_return_history)
    model_GBP = GARCHFXModel(
        initial_parameters=initial_parametes, data=fx2.log_return_history)

    # SET MODEL FOR EACH ASSET.
    stock1.set_model(model_XOM)
    stock2.set_model(model_BOA)
    stock3.set_model(model_SPX)

    fx1.set_model(model_EUR)
    fx2.set_model(model_GBP)

    # CREATE PORTFOLIO AND RUN.
    instruments = [stock1, stock2, stock3]
    holdings = np.array([5, 5, 10000000])

    portfolio = Portfolio(instruments=instruments,
                          init_value=1e7, holdings=holdings)

    portfolio.fit_models()
    portfolio.fit_portfolio()
    # for _ in range(5):
    #     simultion_matrix = portfolio.run_simulation(dependency=True)
    # for row in simultion_matrix:
    #     plt.plot(row)
    #     plt.show()

    temp = portfolio.run_simulation_return_distribution(
        52, 10000, dependency=True)

    ret = []
    for scenario in temp:
        # print(scenario/portfolio.value-1)
        ret.append(scenario/portfolio.value-1)

    plt.hist(ret, bins=25)
    plt.show()

    risk_calc = RiskAssessor(np.array(ret))
    risk_calc.risk_summary()


if __name__ == "__main__":
    main()
