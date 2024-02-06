# -*- coding: utf-8 -*-
import numpy as np

from src.abacus.utils.distributions import gp


def test_pdf_positivity_gpd():
    gp_pdf = gp.pdf
    space = np.linspace(0, 10, 100)

    for x in space:
        assert gp_pdf(x, 0.15, 0.25) >= 0
