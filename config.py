# -*- coding: utf-8 -*-
import pyvinecopulib as pv

EPSILON = 1e-16
DEFALUT_SIMULATIONS = 100
DEFALUT_STEPS = 10
VINE_COPULA_FAMILIES = [pv.BicopFamily.gaussian,
                        pv.BicopFamily.clayton,
                        pv.BicopFamily.frank,
                        pv.BicopFamily.gumbel,
                        pv.BicopFamily.student,
                        pv.BicopFamily.bb1,
                        pv.BicopFamily.bb6,
                        pv.BicopFamily.bb7,
                        pv.BicopFamily.bb8]
