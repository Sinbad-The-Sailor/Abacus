# -*- coding: utf-8 -*-
from enum import Enum


class EquityModel(Enum):
    GARCHNormal = (0,)
    GJRGARCHNormal = (1,)
    GARCHNormalPoission = (2,)
