# -*- coding: utf-8 -*-
from enum import Enum, auto


class OptimizationSpecifications(Enum):
    SP_MAXIMIZE_UTILITY = "sp_maximize_utility.mod"
    MPC_MAXIMIZE_UTILITY = "mpc_maximize_utility.mod"

class DataTypes(Enum):
    LOG_RETURNS = auto(),
    ART_RETURNS = auto(),
    PRICES = auto(),
