# -*- coding: utf-8 -*-
import numpy as np

from abacus.utilities import (
    generalized_pareto,
    norm_poisson_mixture,
    student_poisson_mixture,
)


def test_pdf_positivity_gpd():
    gp_pdf = generalized_pareto.gp.pdf
    space = np.linspace(0, 10, 100)

    for x in space:
        assert gp_pdf(x, 0.15, 0.25) >= 0


def test_pdf_positivity_npm():
    npm_pdf = norm_poisson_mixture.npm.pdf
    space = np.linspace(-100, 100, 1000)

    for x in space:
        assert npm_pdf(x, 0, 1, 0.001, 0.000001) >= 0


def test_pdf_positivity_spm():
    spm_pdf = student_poisson_mixture.spm.pdf
    space = np.linspace(-100, 100, 1000)

    for x in space:
        assert spm_pdf(x, 0, 1, 0.001, 0.000001, 5) >= 0
