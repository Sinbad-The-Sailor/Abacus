import pyvinecopulib as pv

DEFALUT_SIMULATIONS = 100
VINE_COPULA_FAMILIES = [pv.BicopFamily.gaussian,
                        pv.BicopFamily.clayton,
                        pv.BicopFamily.frank,
                        pv.BicopFamily.gumbel,
                        pv.BicopFamily.student,
                        pv.BicopFamily.bb1,
                        pv.BicopFamily.bb6,
                        pv.BicopFamily.bb7,
                        pv.BicopFamily.bb8]
