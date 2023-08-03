# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceHistory:
    mid_history: pd.DataFrame

    @property
    def log_returns(self) -> pd.DataFrame:
        # return np.log(self.mid_history / self.mid_history.shift(1))[1:]
        return np.log(1 + self.mid_history.pct_change())[1:]

    @property
    def art_returns(self) -> pd.DataFrame:
        return self.mid_history.pct_change()[1:]
