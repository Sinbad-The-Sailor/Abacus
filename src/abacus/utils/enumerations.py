# -*- coding: utf-8 -*-
from enum import Enum


class OptimizationSpecifications(Enum):
    SP_MAXIMIZE_UTILITY = "sp_maximize_utility.mod"
    SP_MAXIMIZE_GAIN = "sp_maximize_gain.mod"
    MPC_MAXIMIZE_UTILITY = "mpc_maximize_utility.mod"
