# -*- coding: utf-8 -*-
import pyvinecopulib as pv

# Years and Months
MINIMUM_PRICE_HISTORY = (1, 0)

# Processes
MAXIMUM_AR_ORDER = 5
INITIAL_VARIANCE_GARCH_OBSERVATIONS = 20
INITIAL_GARCH_PARAMETERS = (0.05, 0.90)
STOCK_ADMISSIBLE_MODELS = (
    "AR",
    "GARCH"
)
VINE_COPULA_FAMILIES = (
    pv.BicopFamily.gaussian,
    pv.BicopFamily.clayton,
    pv.BicopFamily.frank,
    pv.BicopFamily.gumbel,
    pv.BicopFamily.student,
    pv.BicopFamily.bb1,
    pv.BicopFamily.bb6,
    pv.BicopFamily.bb7,
    pv.BicopFamily.bb8,
)

# Computations
VINE_COPULA_NUMBER_OF_THREADS = 6

# Optimizer
DEFAULT_SOLVER = "ipopt"
