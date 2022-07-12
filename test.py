import numpy as np

from test_models import GARCHEquityModel, GARCHFXModel
from test_instruments import Equity, FX, Instrument

SIMULATIONS = 1e3


class Portfolio():

    instruments: list[Instrument]

    def __init__(self, instruments):
        self.instruments = instruments
        try:
            self.number_of_instruments = len(instruments)
        except ValueError:
            self.number_of_instruments = 0

    def fit_models(self):

        # Check if all instruments has a model.
        if not self._has_models():
            raise ValueError(f"One instrument has no model.")

        # Call all fit models.
        for instrument in self.instruments:
            instrument.model.fit_model()

    def run_simulation(self, number_of_iterations: int = SIMULATIONS, dependency: bool = True) -> np.array:

        # Check if portfolio has instruments.
        if not self._has_instruments():
            raise ValueError("Portfolio has no instruments.")

        # Check if all instruments has a model.
        if not self._has_models():
            raise ValueError("One instrument has no model.")

        if dependency:
            return self._generate_multivariate_simulation(number_of_iterations=number_of_iterations)
        else:
            return self._generate_univariate_simulation(number_of_iterations=number_of_iterations)

    def _has_models(self):
        for instrument in self.instruments:
            if instrument.has_model == False:
                return False
        return True

    def _has_instruments(self):
        if self.number_of_instruments == 0:
            return False
        return True

    def _generate_univariate_simulation(self, number_of_iterations: int):
        number_of_assets = len(self)

    def _generate_multivariate_simulation(self, number_of_iterations: int):
        number_of_assets = len(self)

    def __len__(self):
        return self.number_of_instruments


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

    portfolio.fit_models()
    portfolio.run_simulation()


if __name__ == "__main__":
    main()
