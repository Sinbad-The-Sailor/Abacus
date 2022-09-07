# -*- coding: utf-8 -*-
class GJRGARCHModel(Model):
    def __init__(self, data):
        super().__init__(data, initial_parameters=INITIAL_GJRGARCH_PARAMETERS)
        self.last_volatility_estimate = 0
        self.volatility_sample = None
        self.inital_volatility_esimate = np.std(self.data[:20])
        self.long_run_volatility_estimate = np.std(self.data)
        self.log_likelihood_value = None
        self.number_of_parameters = 3

    def run_simulation(self, number_of_steps: int = DEFALUT_STEPS) -> np.array:
        result = self._generate_simulation(number_of_steps=number_of_steps, isVol=False)
        return result

    def run_volatility_simulation(
        self, number_of_steps: int = DEFALUT_STEPS
    ) -> np.array:
        result = self._generate_simulation(number_of_steps=number_of_steps, isVol=True)
        return result

    def generate_uniform_samples(self) -> np.array:
        result = np.zeros(self.number_of_observations - 1)

        # Check if a solution exists.
        if not self._has_solution():
            raise NoParametersError

        # Check if a volatility estimate exists.
        if self.volatility_sample is None:
            self.volatility_sample = self._generate_volatility(self.optimal_parameters)

        # Create normalized sample and transform it in one go.
        for i in range(1, self.number_of_observations):
            # TODO: REMOVE -1. Make all arrays have correct lenght.
            normalized_sample = self.data[i] / self.volatility_sample[i]
            uniform_sample = norm.cdf(normalized_sample, loc=0, scale=1)
            result[i - 1] = uniform_sample

        return result

    def generate_correct_samples(self, uniform_samples: np.array) -> np.array:

        # Create volatility samples.
        number_of_observations = len(uniform_samples)
        volatility_samples = self.run_volatility_simulation(
            number_of_steps=number_of_observations
        )

        # Initialize empty numpy array.
        result = np.zeros(number_of_observations)

        # Transform samples and unnormalize in one go.
        for i in range(0, number_of_observations):
            uniform_sample = uniform_samples[i]
            normal_sample = norm.ppf(uniform_sample, loc=0, scale=1)
            result[i] = normal_sample * volatility_samples[i]

        return result

    def fit_model(self) -> bool:
        # TODO: Add number of iterations and while loop.

        initial_parameters = self._precondition_parameters(self.initial_parameters)

        solution = minimize(self._cost_function, initial_parameters, args=self.data)
        self.optimal_parameters = solution.x
        self.last_volatility_estimate = self._generate_volatility(
            self.optimal_parameters
        )[-1]
        self.log_likelihood_value = solution.fun
        print(
            f" {self._uncondition_parameters(self.optimal_parameters)} {solution.success} {solution.fun}"
        )
        if not solution.success:
            logger.warning("minimizer not succesful.")
        return solution.success

    def _cost_function(self, params: np.array, data: np.array) -> float:
        vol_est = self._generate_volatility_squared(params)
        log_loss = np.sum(np.log(vol_est) + (data ** 2) / vol_est)
        return log_loss

    def plot_volatility(self):
        if not self._has_solution():
            raise NoParametersError
        params = self.optimal_parameters
        vol_result = self._generate_volatility(params=params)
        plt.plot(vol_result)
        plt.show()

    def _generate_volatility(self, params: np.array) -> np.array:
        result = np.sqrt(self._generate_volatility_squared(params=params))
        return result

    def _generate_volatility_squared(self, params: np.array) -> np.array:
        result = np.zeros(self.number_of_observations)
        for i in range(0, self.number_of_observations):
            if i == 0:
                result[i] = self.inital_volatility_esimate ** 2
            else:
                mu_corr = np.exp(-np.exp(-params[0]))
                mu_ewma = np.exp(-np.exp(-params[1]))
                mu_asym = np.exp(-np.exp(-params[2]))

                result[i] += self.long_run_volatility_estimate ** 2 + mu_corr * (
                    mu_asym * result[i - 1]
                    + (1 - mu_ewma) * self.data[i - 1] ** 2
                    + 2
                    * (mu_ewma - mu_asym)
                    * self.data[i - 1] ** 2
                    * np.where(self.data[i - 1] < 0, 1, 0)
                    - self.long_run_volatility_estimate ** 2
                )

        return result

    def _generate_simulation(self, number_of_steps: int, isVol: bool) -> np.array:

        # Check if optimal parameters exist.
        if not self._has_solution():
            raise ValueError("Model has no fitted parameters.")

        # Check if initial volatility exist.
        if self.last_volatility_estimate == 0:
            raise ValueError("Model has no initial volatility estimate.")

        # Initialize empty numpy array.
        return_result = np.zeros(number_of_steps)
        volatility_result = np.zeros(number_of_steps)

        # Inital paramters for reursion start.
        return_estimate = self.data[-1]
        volatility_estimate = self.last_volatility_estimate

        parameters = self._uncondition_parameters(self.optimal_parameters)
        alpha = parameters[0]
        beta = parameters[1]
        gamma = parameters[2]
        omega = self.long_run_volatility_estimate ** 2 * (1 - alpha - beta)

        # Generation of return estimates.
        for i in range(number_of_steps):
            sample = norm.rvs(size=1, loc=0, scale=1)
            volatility_estimate = np.sqrt(
                omega
                + beta * volatility_estimate ** 2
                + (alpha + gamma * np.where(return_estimate < 0, 1, 0))
                * return_estimate ** 2
            )
            return_estimate = sample * volatility_estimate

            return_result[i] = return_estimate
            volatility_result[i] = volatility_estimate

        if isVol:
            return volatility_result
        else:
            return return_result

    @staticmethod
    def _precondition_parameters(params: np.array) -> np.array:
        mu_corr = params[0] + params[1] + params[2]
        mu_ewma = (params[1] + 0.5 * params[2]) / (
            params[0] + params[1] + 0.5 * params[2]
        )
        mu_asym = params[1] / (params[0] + params[1] + 0.5 * params[2])

        z_corr = np.log(-1 / np.log(mu_corr))
        z_ewma = np.log(-1 / np.log(mu_ewma))
        z_asym = np.log(-1 / np.log(mu_asym))

        return np.array([z_corr, z_ewma, z_asym])

    @staticmethod
    def _uncondition_parameters(params: np.array) -> np.array:
        mu_corr = np.exp(-np.exp(-params[0]))
        mu_ewma = np.exp(-np.exp(-params[1]))
        mu_asym = np.exp(-np.exp(-params[2]))

        alpha = mu_corr * (1 - mu_ewma)
        beta = mu_corr * mu_asym
        gamma = 2 * mu_corr * (mu_ewma - mu_asym)

        return np.array([alpha, beta, gamma])
