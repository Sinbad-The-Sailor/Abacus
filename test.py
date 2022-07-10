import numpy as np

from test_models import GARCHEquityModel, GARCHFXModel
from test_instruments import Equity, FX


class Portfolio():

    instruments: list

    def __init__(self, instruments):
        self.instruments = instruments

    def fit_models(self):

        # Check if all instruments has a model.
        self._has_models()

        # Call all fit models.
        for instrument in self.instruments:
            instrument.model.fit_model()

    def run_simulation(self) -> np.array:

        # Check if all instruments has a model.
        self._has_models()

        # TODO: Apply appropriate vine copula to all assets.

        # Run simulations.
        for instrument in self.instruments:
            instrument.model.run_simulation(1000)

    def _has_models(self):
        for instrument in self.instruments:
            if instrument.has_model == False:
                raise ValueError(f"Instrument {instrument} has no model.")


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
    initial_parametes = [0.00001, 0.2, 0.70]

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
    print("MODEL")
    stock1.model.fit_model()
    stock1.model.plot()


if __name__ == "__main__":
    main()
