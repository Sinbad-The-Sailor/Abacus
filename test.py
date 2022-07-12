import numpy as np
import copulae
import pyvinecopulib


from test_models import GARCHEquityModel, GARCHFXModel
from test_instruments import Equity, FX, Instrument

SIMULATIONS = 1000


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

        # Check if all models are fitted.
        if not self._has_solution():
            raise ValueError("One model has no solution.")

        if dependency:
            return self._generate_multivariate_simulation(number_of_iterations=number_of_iterations)
        else:
            return self._generate_univariate_simulation(number_of_iterations=number_of_iterations)

    def _generate_univariate_simulation(self, number_of_iterations: int) -> np.array:
        # TODO: Remove list to make this faster!
        result = []

        for instrument in self.instruments:
            result.append(instrument.model.run_simulation(
                number_of_iterations=number_of_iterations))

        return np.vstack(result)

    def _generate_multivariate_simulation(self, number_of_iterations: int) -> np.array:

        # Check if more than 1 asset exists.
        if self.number_of_instruments == 1:
            raise ValueError("To few instruments to run dependency.")

        # Creating uniform data.
        # TODO: Remove list to make this faster!
        uniforms = []

        if self.number_of_instruments == 2:
            # TODO: Function which picks optimal vanilla copula from a family of copulae.
            copula = copulae.StudentCopula()

            pass

        if self.number_of_instruments > 2:
            # Function which picks optimal vine copula.

            pass

        counter = 0
        for instrument in self.instruments:
            if counter < 2:
                uniforms.append(instrument.model.generate_uniform_samples())
                print(len(instrument.model.generate_uniform_samples()))
            counter = counter + 1
        uniform_matrix = np.vstack(uniforms)
        for row in uniform_matrix.T:
            print(row)

        student_copula = copulae.StudentCopula()
        student_copula.fit(uniform_matrix.T)
        print(student_copula.params[1][0])

        result = []

    def _has_models(self) -> bool:
        for instrument in self.instruments:
            if instrument.has_model == False:
                return False
        return True

    def _has_instruments(self) -> bool:
        if self.number_of_instruments == 0:
            return False
        return True

    def _has_solution(self) -> bool:
        for instrument in self.instruments:
            if instrument.model._has_solution == False:
                return False
        return True

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
    from matplotlib import pyplot as plt
    portfolio.fit_models()

    simultion_matrix = portfolio.run_simulation(dependency=True)

    # for row in simultion_matrix:
    # plt.plot(row)
    #   plt.show()


if __name__ == "__main__":
    main()
