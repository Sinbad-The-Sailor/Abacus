from test_models import GARCHEquityModel, GARCHFXModel
from test_instruments import Equity, FX
from test_portfolio import Portfolio


def main():

    # CREATE ASSETS.
    start = "2011-12-28"
    end = "2021-12-28"
    interval = "wk"

    stock1 = Equity(ric="XOM", currency="USD", start_date=start,
                    end_date=end, interval=interval)
    stock2 = Equity(ric="GS", currency="USD", start_date=start,
                    end_date=end, interval=interval)

    fx1 = FX(ric="USDEUR=X", currency="USD", start_date=start,
             end_date=end, interval=interval)
    fx2 = FX(ric="USDGBP=X", currency="USD", start_date=start,

             end_date=end, interval=interval)

    # CREATE MODELS FOR EACH ASSET.
    initial_parametes = [0.01, 0.01, 0.7]

    model_XOM = GARCHEquityModel(
        initial_parameters=initial_parametes, data=stock1.return_history)
    model_BOA = GARCHEquityModel(
        initial_parameters=initial_parametes, data=stock2.return_history)

    model_EUR = GARCHFXModel(
        initial_parameters=initial_parametes, data=fx1.return_history)
    model_GBP = GARCHFXModel(
        initial_parameters=initial_parametes, data=fx2.return_history)

    # SET MODEL FOR EACH ASSET.
    stock1.set_model(model_XOM)
    stock2.set_model(model_BOA)

    fx1.set_model(model_EUR)
    fx2.set_model(model_GBP)

    # CREATE PORTFOLIO AND RUN.
    instruments = [stock1, stock2, fx1, fx2]

    portfolio = Portfolio(instruments=instruments)
    from matplotlib import pyplot as plt
    portfolio.fit_models()

    simultion_matrix = portfolio.run_simulation(dependency=True)
    for row in simultion_matrix:
        plt.plot(row)
        plt.show()


if __name__ == "__main__":
    main()
