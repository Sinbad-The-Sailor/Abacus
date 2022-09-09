# -*- coding: utf-8 -*-
import numpy as np
import pyvinecopulib as pv

EPSILON = 1e-8
DEFALUT_SIMULATIONS = 100
DEFALUT_STEPS = 10
VINE_COPULA_FAMILIES = [
    pv.BicopFamily.gaussian,
    pv.BicopFamily.clayton,
    pv.BicopFamily.frank,
    pv.BicopFamily.gumbel,
    pv.BicopFamily.student,
    pv.BicopFamily.bb1,
    pv.BicopFamily.bb6,
    pv.BicopFamily.bb7,
    pv.BicopFamily.bb8,
]
INITIAL_GARCH_PARAMETERS = np.array([0.05, 0.80])
INITIAL_GJRGARCH_PARAMETERS = np.array([0.05, 0.80, 0.001])
INITIAL_NPM_PARAMETERS = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

ADMISSIBLE_EQUTIY_MODELS = {"AR": [1, 2],
                            "MA": [1, 2],
                            "NNAR": [1, 2],
                            "GARCH": [],
                            "GJRGARCH": []}
ADMISSIBLE_FX_MODELS  = {"AR": [1, 2],
                         "NNAR": [1, 2],
                         "GARCH": []}
