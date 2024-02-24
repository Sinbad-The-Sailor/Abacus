# -*- coding: utf-8 -*-
import pyvinecopulib as pv



# Processes
MAXIMUM_AR_ORDER = 5
INITIAL_VARIANCE_GARCH_OBSERVATIONS = 20
INITIAL_GARCH_PARAMETERS = 0.05, 0.90

# Copulas
VINE_COPULA_NUMBER_OF_THREADS = 6
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

# Extreme Value Theory POT Threshold
EVT_THRESHOLD = 0.95
GEV_INITIAL_SOLUTION = 0.1, 0.01

# Optimizer
DEFAULT_SOLVER = "ipopt"
