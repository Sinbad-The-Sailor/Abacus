# An equity model can be thought of BEING a model of a STOCK itself.
# The stock will have some price history, ticker, bid-ask spreads, etc, currency, ISIN
# This data will primarily exist in the StockData class, which will be fed into the StockModel for further usage

# When creating an equity model you will feed it data from a stock_data object.
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm

from scipy.optimize import minimize

from distributions.normal_poisson_mixture import norm_poisson_mix_pdf, student_poisson_mix_pdf, student_poisson_mix_cdf, \
    norm_poisson_mix_cdf


class StockData:
    # List of data in ascending order! Since dicts are hash-tables one might think about using dicts with
    # dates as a key

    def __init__(self, ric):
        self.adj_close = pd.DataFrame()
        self.ric = ric

    def get_log_returns(self):
        """
        Calculated logarithmic returns of adjusted close prices.

        Returns: List of adjusted close logarithmic returns.
        """

        return np.log(self.adj_close / self.adj_close.shift(1))[1:]


class EquityModel:
    _stock_data = None
    _model_solution = None
    _model_fitted = False
    _uniform_transformed_returns = []

    def __init__(self, stock_data):
        self._stock_data = stock_data

    def run_simulation(self):
        pass

    def fit_model(self, model='normal'):
        """

        Args:
            model:

        Returns:

        """
        data = self._stock_data.get_log_returns()
        data = data.to_list()

        if model == 'normal':
            cons = self._likelihood_constraints_normal()
            func = self._likelihood_function_normal

            # Initial conditions for GJR-GARCH(1,1) model.
            omega0 = 0.001
            alpha0 = 0.05
            beta0 = 0.80
            beta1 = 0.02
            mu0 = np.mean(data)
            x0 = [omega0, alpha0, beta0, beta1, mu0]

            garch_model_solution = minimize(func, x0, constraints=cons, args=data)

            # Added to keep method non-static. Might be useful to keep in Equity Model object instead of re-running
            # the optimization.
            self._model_solution = garch_model_solution
            self._model_fitted = True

            return garch_model_solution

        elif model == 'normal poisson mixture':
            cons = self._likelihood_constraints_normal_poisson_mix()
            func = self._likelihood_function_normal_poisson_mixture

            # Initial conditions for GJR-GARCH(1,1) model with Poisson normal jumps.
            omega0 = 0.001
            alpha0 = 0.05
            beta0 = 0.80
            beta1 = 0.02
            mu0 = np.mean(data)
            kappa = 0.05
            lamb = 0.05
            x0 = [omega0, alpha0, beta0, beta1, mu0, kappa, lamb]
            garch_poisson_model_solution = minimize(func, x0, constraints=cons, args=data)

            self._model_solution = garch_poisson_model_solution
            self._model_fitted = True
            ##REMOVE
            print(self._model_solution)

            return garch_poisson_model_solution

        elif model == 'student poisson mixture':
            cons = self._likelihood_constraints_student_poisson_mix()
            func = self._likelihood_function_student_poisson_mixture

            # Initial conditions for GJR-GARCH(1,1) model with Poisson normal jumps.
            omega0 = 0.001
            alpha0 = 0.05
            beta0 = 0.80
            beta1 = 0.02
            mu0 = np.mean(data)
            kappa = 2.9
            lamb = 0.1587
            nu = 7
            x0 = [omega0, alpha0, beta0, beta1, mu0, kappa, lamb, nu]
            garch_poisson_model_solution = minimize(func, x0, constraints=cons, args=data)

            self._model_solution = garch_poisson_model_solution
            self._model_fitted = True

            ##REMOVE
            print(self._model_solution)

            return garch_poisson_model_solution

        elif model == 'generalized hyperbolic':
            cons = self._likelihood_constraints_generalized_hyperbolic()
            func = self._likelihood_constraints_generalized_hyperbolic

    # noinspection PyMethodMayBeStatic
    def _likelihood_function_normal(self, params, data) -> float:
        """
        Conditional GJR-GARCH likelihood function with normal innovations, derived as in John C. Hull Risk Management.
        Note, the vol_estimate is squared to simplify calculations. Note, the negative log likelihood is return to be
        minimized.

        :param params: list of parameters involved. By standard notation it follows:
                       param[0] is omega |
                       param[1] is alpha |
                       param[2] is beta0 |
                       param[3] is beta1 (asymmetry modifier) |
                       param[4] is mu
        :param data: list of observations, e.g. log-returns over time.
        :return: Negative log-likelihood value.
        """
        n_observations = len(data)
        log_likelihood = 0
        initial_squared_vol_estimate = (params[0]
                                        + params[1] * (data[0] ** 2)
                                        + params[3] * (data[0] ** 2) * np.where(data[0] < 0, 1, 0)
                                        + params[2] * (data[0] ** 2))
        current_squared_vol_estimate = initial_squared_vol_estimate

        for i in range(1, n_observations):
            log_likelihood = (log_likelihood
                              + ((data[i-1] - params[4]) ** 2) / current_squared_vol_estimate
                              + 2 * np.log(np.sqrt(current_squared_vol_estimate)))

            current_squared_vol_estimate = (params[0] + params[1] * (data[i-1] ** 2)
                                            + params[3] * (data[i-1] ** 2) * np.where(data[i-1] < 0, 1, 0)
                                            + params[2] * current_squared_vol_estimate)

        return log_likelihood

    # noinspection PyMethodMayBeStatic
    def _likelihood_constraints_normal(self) -> dict:
        """

        :return:
        """
        # x[0] is omega
        # x[1] is alpha
        # x[2] is beta0
        # x[3] is beta1 (asymmetry modifier)

        cons_garch = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] - (0.5 * x[3]) + 1},
                      {'type': 'ineq', 'fun': lambda x:  x[0]},
                      {'type': 'ineq', 'fun': lambda x:  x[1] + x[3]},
                      {'type': 'ineq', 'fun': lambda x:  x[2]}]
        return cons_garch

    # noinspection PyMethodMayBeStatic
    def _likelihood_function_normal_poisson_mixture(self, params, data):
        """

        :param params:
        :param data:
        :return:
        """
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        # param[5] is kappa
        # param[6] is lambda
        n_observations = len(data)
        log_likelihood = 0
        initial_squared_vol_estimate = (params[0]
                                        + params[1] * (data[0] ** 2)
                                        + params[3] * (data[0] ** 2) * np.where(data[0] < 0, 1, 0)
                                        + params[2] * (data[0] ** 2))
        current_squared_vol_estimate = initial_squared_vol_estimate

        for i in range(0, n_observations):
            log_likelihood = log_likelihood + np.log(norm_poisson_mix_pdf(data[i], params[4],
                                                     np.sqrt(current_squared_vol_estimate), params[5], params[6]))

            current_squared_vol_estimate = (params[0] + params[1] * (data[i - 1] ** 2)
                                            + params[3] * (data[i - 1] ** 2) * np.where(data[i - 1] < 0, 1, 0)
                                            + params[2] * current_squared_vol_estimate)

        return -log_likelihood

    # noinspection PyMethodMayBeStatic
    def _likelihood_constraints_normal_poisson_mix(self) -> dict:
        """

        :return:
        """
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        # param[5] is kappa
        # param[6] is lambda
        cons_garch_poisson = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] - (0.5 * x[3]) + 1},
                              {'type': 'ineq', 'fun': lambda x: x[0]},
                              {'type': 'ineq', 'fun': lambda x: x[1] + x[3]},
                              {'type': 'ineq', 'fun': lambda x: x[2]},
                              {'type': 'ineq', 'fun': lambda x: x[5]},
                              {'type': 'ineq', 'fun': lambda x: x[6]}
                              ]
        return cons_garch_poisson

    # noinspection PyMethodMayBeStatic
    def _likelihood_function_student_poisson_mixture(self, params, data):
        """

        :param params:
        :param data:
        :return:
        """
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        # param[5] is kappa
        # param[6] is lambda
        # param[7] is nu
        n_observations = len(data)
        log_likelihood = 0
        initial_squared_vol_estimate = (params[0]
                                        + params[1] * (data[0] ** 2)
                                        + params[3] * (data[0] ** 2) * np.where(data[0] < 0, 1, 0)
                                        + params[2] * (data[0] ** 2))
        current_squared_vol_estimate = initial_squared_vol_estimate

        for i in range(0, n_observations):
            log_likelihood = log_likelihood + np.log(student_poisson_mix_pdf(data[i], params[4],
                                                                             np.sqrt(current_squared_vol_estimate),
                                                                             params[5], params[6], params[7]
                                                                             ))

            current_squared_vol_estimate = (params[0] + params[1] * (data[i - 1] ** 2)
                                            + params[3] * (data[i - 1] ** 2) * np.where(data[i - 1] < 0, 1, 0)
                                            + params[2] * current_squared_vol_estimate)

        return -log_likelihood

    # noinspection PyMethodMayBeStatic
    def _likelihood_constraints_student_poisson_mix(self) -> dict:
        """

        :return:
        """
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        # param[5] is kappa
        # param[6] is lambda
        # param[7] is nu
        abstol = 1e-6
        cons_garch_poisson = [{'type': 'ineq', 'fun': lambda x: -x[1] - x[2] - (0.5 * x[3]) + 1},
                              {'type': 'ineq', 'fun': lambda x: x[0]-abstol},
                              {'type': 'ineq', 'fun': lambda x: x[1] + x[3]-abstol},
                              {'type': 'ineq', 'fun': lambda x: x[2]-abstol},
                              {'type': 'ineq', 'fun': lambda x: x[5]-abstol},
                              {'type': 'ineq', 'fun': lambda x: x[6]-abstol},
                              {'type': 'ineq', 'fun': lambda x: x[7]-3-abstol}]
        return cons_garch_poisson

    def plot_volatility(self):
        # Plotting GJR-GARCH fitted volatility.
        params = self._model_solution.x
        omg = params[0]
        alp = params[1]
        bet = params[2]
        gam = params[3]

        data = self._stock_data.get_log_returns()
        time = data.index[1:]
        data = data.to_list()
        vol = []

        curr_vol = (omg
                    + alp * (data[0] ** 2)
                    + gam * (data[0] ** 2) * np.where(data[0], 1, 0)
                    + bet * (data[0] ** 2))
        vol.append(curr_vol)

        for i in range(2, len(data)):
            curr_vol = (omg
                        + alp * (data[i - 1] ** 2)
                        + gam * (data[i - 1] ** 2) * np.where(data[i - 1] < 0, 1, 0)
                        + bet * (vol[i - 2]))
            vol.append(curr_vol)

        vol = np.sqrt(vol)
        plt.title(self._stock_data.ric)
        plt.plot(time[20:], vol[20:])
        plt.show()

    def _generate_uniform_return_observations(self):
        # param[0] is omega
        # param[1] is alpha
        # param[2] is beta0
        # param[3] is beta1 (asymmetry modifier)
        # param[4] is mu
        # param[5] is kappa
        # param[6] is lambda
        # param[7] is nu

        uniforms = []
        data = self._stock_data.get_log_returns()
        params = self._model_solution.x
        curr_vol = (params[0]
                    + params[1] * (data[0] ** 2)
                    + params[3] * (data[0] ** 2) * np.where(data[0], 1, 0)
                    + params[2] * (data[0] ** 2))
        for i in range(1, len(data)):
            u = student_poisson_mix_cdf(data[i], params[4], np.sqrt(curr_vol), params[5], params[6], params[7])
            print(u)
            u = norm.ppf(u, 0, 1)

            # When using normal, apply this to find quantile-quantile plot.
            # u = (data[i] - params[4]) / np.sqrt(curr_vol)
            # u = norm.cdf(u, 0, 1)
            uniforms.append(u)
            curr_vol = (params[0]
                        + params[1] * (data[i] ** 2)
                        + params[3] * (data[i] ** 2) * np.where(data[i], 1, 0)
                        + params[2] * (curr_vol ** 2))
        return np.array(uniforms)

    def plot_qq(self):
        uniforms = self._generate_uniform_return_observations()
        # print(uniforms)
        # uniforms = np.array([0.2355308106743407, 1.0123330201868257, 0.39007099401279804, 0.8018236669397528
        # , 0.7197739224659218,
        #  0.3089332968440605, 0.9467258073257198, 0.6691295789643134, 0.14409443323794693, 0.3395247481945062,
        #  0.6477521646516081, 0.09830541061569531, 0.7251616289032655, 0.9963219669549799, 0.9357035053919509,
        #  0.8797167372693029, 1.0088739527878925, 0.4059636809638224, 0.5766043484666629, 0.09714799142760418,
        #  0.25127672065851625, 1.172366772963457, 0.23164595293810397, 1.022380253386123, 0.8489397458938767])
        # qqplot(uniforms, norm)

        qqplot(uniforms, norm, fit=False, line='q')
        plt.show()

v = np.array([0.31502940002470026,
0.6461011732674571,
0.6930674781020274,
0.10066724188067759,
0.32211334480289217,
0.8129665265388029,
0.6678194025833366,
0.7954859460507077,
0.6034362997915744,
0.6404191412874132,
0.3375967727713593,
0.7017698964715164,
0.266371568089098,
0.6763374204157702,
0.5161910352481488,
0.2591895344675145,
0.3761550302580213,
0.3585137409666113,
0.47860224714229993,
0.7574157576761051,
0.4369683841490727,
0.4128778842094703,
0.2676953393038529,
0.21717685058263814,
0.8422547494190585,
0.6266711064584416,
0.6596703661959387,
0.5305330645642395,
0.4252329845333306,
0.5015796902238959,
0.44362018447398377,
0.43401224036811337,
0.334692151752961,
0.513625030913498,
0.1459596487188792,
0.5177233657596502,
0.6177831935253714,
0.4610936295291792,
0.3359593813703336,
0.29628627515264494,
0.4949211055741979,
0.33339841876961007,
0.18544856421835482,
0.880264178990374,
0.5577891491101195,
0.6528041300018952,
0.29693618354342777,
0.8331287970426978,
0.427816755940363,
0.4745526296845551,
0.27812385137147516,
0.6877640034076251,
0.42030186383676127,
0.4127865305952362,
0.5937448399300875,
0.4538410734488104,
0.5611565602302941,
0.6627381316303217,
0.541750539397011,
0.5386327171771889,
0.3688330428903653,
0.4722271867045387,
0.23728953827581678,
0.558688960690842,
0.3892753866434372,
0.27482379155514364,
0.5661525979727671,
0.6624669610615189,
0.5330960457009847,
0.38235325556981364,
0.37723240605558417,
0.6516468611612533,
0.5832071619945249,
0.3900516424179994,
0.44401452434351335,
0.46807961335568954,
0.34434539225423966,
0.44661441295383636,
0.6481428053098953,
0.5190208483153602,
0.4325788253928217,
0.4649080175794683,
0.2381083499813065,
0.3404635804934941,
0.5470515671624206,
0.6430911589533248,
0.34531895485672387,
0.6713609473666762,
0.4479699080589361,
0.5415252858491949,
0.6171605905231562,
0.45894096565131426,
0.5687786057781352,
0.4706456254453417,
0.47597516025145364,
0.40254675889581165,
0.4147691908432665,
0.5764437163247184,
0.6647757257209577,
0.4899673673666462,
0.576846029730908,
0.4790697665099646,
0.3802369306349758,
0.7909053449402679,
0.6674391356350421,
0.5683065931207694,
0.7513009234914244,
0.06731703115136843,
0.03612902406958661,
0.89232719004439,
0.5205109337485883,
0.17589165988945968,
0.8641063071687293,
0.2694051504909276,
0.02077357602387624,
0.7077050881647182,
0.24754782611888623,
0.7237879350200925,
0.5161132353483859,
0.43398713872801215,
0.3981784889471118,
0.7720635575594829,
0.3570678314284766,
0.483156421392975,
0.511167916520755,
0.6778018638067967,
0.4393492810086349,
0.30560001223183997,
0.2513310214526394,
0.6636803622338157,
0.660524949418921,
0.4387204067316253,
0.5293854700064177,
0.5535378303410075,
0.39747291382176325,
0.5275841144683854,
0.5684677742315263,
0.5791460820171552,
0.288186069606596,
0.6125672357271963,
0.5664891951070854,
0.3553946684366886,
0.2947362371280793,
0.05673699417995043,
0.44287121421517417,
0.0616437309120319,
0.7619585198779945,
0.7368280417935943,
0.22204065973576606,
0.0684380045668857,
0.9223479675887805,
0.05143381152243011,
0.2738374545816217,
0.01185696226075489,
0.7805551273458794,
0.7034275859674778,
0.7825711968307755,
0.8111179667318117,
0.4040210130302455,
0.6702248262008931,
0.44346342326407084,
0.781022379086483,
0.5302808306147133,
0.4966341025103616,
0.16487789953559556,
0.8145531880648827,
0.3265647084275122,
0.6190251583168137,
0.7315352468554234,
0.5139874133458603,
0.4227316838282064,
0.6193444959219676,
0.46582773629952356,
0.1631792336042644,
0.32638997928761054,
0.26997226757237025,
0.13000814716350484,
0.9082642566225625,
0.5091562987038359,
0.7484270122789859,
0.3922300523435669,
0.6818401182763448,
0.555834226096017,
0.2622877505373652,
0.6811297864583601,
0.10055450481702415,
0.3708510654851284,
0.2879454459832859,
0.23827752777073669,
0.8069721659766826,
0.6945992085308176,
0.5829827142745261,
0.3592366787110231,
0.2898853510158813,
0.38549640932790474,
0.5313518086034282,
0.5185553015397677,
0.6212372068759768,
0.6560440559323787,
0.5668189249203004,
0.571890407088649,
0.3860362122479674,
0.5873531955300468,
0.4597180039213498,
0.5480806715617975,
0.68165907438224,
0.525783365928926,
0.4109333156505262,
0.5807878711412736,
0.721957826369731,
0.2904624677713140,
0.16908154138733403,
0.837168168029563,
0.6666014845593674,
0.2609921256969694,
0.0016810844699455178,
0.5207839731217736,
0.005132380573870971,
0.0013800939971384872,
0.9592292869847008,
0.22923406952395733,
0.9893231812478149,
0.7592048973485848,
0.2585248435707741])
# v = norm.ppf(v, 0, 1)
# qqplot(v, norm)
# plt.show()