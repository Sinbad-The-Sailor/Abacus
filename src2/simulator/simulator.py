# -*- coding: utf-8 -*-

import torch
import pandas as pd

from utils.instruments import Asset

class Simulator:

    def __init__(self, instruments: list[Asset]):
        self._instruments = instruments
        self.is_calibrated = False



    def calibrate(self):
        # Find models for each risk factor.
        # Use a factory design pattern.
        # If none are succesful throw an error.
        ...


    def run_simulation(time_steps: int) -> torch.Tensor:
        # Check for succesful calibration, Throw an error otherwise.
        ...
