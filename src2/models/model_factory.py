# -*- coding: utf-8 -*-

import torch
import pandas as pd

from utils.instruments import RiskFactor

class ModelFactory:

    def __init__(self, risk_factors: list[RiskFactor]):
        self.risk_factors = risk_factors
