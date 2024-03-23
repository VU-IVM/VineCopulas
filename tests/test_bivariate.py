# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:28:28 2024


"""
import warnings

warnings.filterwarnings("ignore")
from vinecopulas.bivariate import *
import numpy as np
import unittest


# %% Gaussian
class TestGaussian(unittest.TestCase):
    def setUp(self):
        self.copula = 1
        self.X = np.array(
            [
                [0.08, 0.11],
                [0.94, 0.94],
                [0.31, 0.23],
                [0.39, 0.4],
                [0.51, 0.49],
                [0.27, 0.41],
            ]
        )
        self.par = 0.95
        self.hfunc1 = np.array(
            [0.63563113, 0.59830531, 0.19555359, 0.51533551, 0.43779767, 0.87196238]
        )
        self.hfunc2 = np.array(
            [0.22118469, 0.59830531, 0.74534146, 0.45075869, 0.56220233, 0.10199219]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.isclose(self.X, ur, rtol=0.05).all()

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.around(alpha, decimals=2) == 0.98

    def test_CDF(self):
        expected_cdf = np.array(
            [0.07056682, 0.92506194, 0.21669348, 0.34607267, 0.44883543, 0.26371613]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [6.39809424, 10.39807426, 2.91290329, 3.30456429, 3.1645501, 1.72445976]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -8.16

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.13506725, 0.79969318, 0.35777419, 0.47770374, 0.50903404, 0.20660369]
        )
        expected_u1 = np.array(
            [0.07343121, 0.79738852, 0.2324708, 0.47843132, 0.46545735, 0.40998779]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(10)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Gumbel0
class TestGumbel0(unittest.TestCase):
    def setUp(self):
        self.copula = 2
        self.X = np.array(
            [
                [0.15, 0.25],
                [0.87, 0.7],
                [0.14, 0.18],
                [0.06, 0.12],
                [0.66, 0.66],
                [0.96, 0.94],
            ]
        )
        self.par = 4
        self.hfunc1 = np.array(
            [0.73276324, 0.04697687, 0.55981062, 0.6614234, 0.54964709, 0.2461972]
        )
        self.hfunc2 = np.array(
            [0.17155244, 0.98089955, 0.2888754, 0.1415532, 0.54964709, 0.87558429]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(4, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.13266769, 0.69856334, 0.11037844, 0.04894353, 0.61009907, 0.93742528]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [2.35483012, 0.61760498, 3.45948346, 3.81506808, 3.50156649, 10.90600741]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -6.6

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.24662572, 0.69422802, 0.18006648, 0.1175218, 0.65959162, 0.94589063]
        )
        expected_u1 = np.array(
            [0.13207903, 0.38675177, 0.16878067, 0.18936538, 0.63741259, 0.93914661]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(10)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Gumbel90


class TestGumbel90(unittest.TestCase):
    def setUp(self):
        self.copula = 3
        self.X = np.array(
            [
                [0.85, 0.25],
                [0.13, 0.7],
                [0.86, 0.18],
                [0.94, 0.12],
                [0.34, 0.66],
                [0.04, 0.94],
            ]
        )
        self.par = 4
        self.hfunc1 = np.array(
            [0.73276324, 0.04697687, 0.55981062, 0.6614234, 0.54964709, 0.2461972]
        )
        self.hfunc2 = np.array(
            [0.82844756, 0.01910045, 0.7111246, 0.8584468, 0.45035291, 0.12441571]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(4, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.11733231, 0.00143666, 0.06962156, 0.07105647, 0.04990093, 0.00257472]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [2.35483012, 0.61760498, 3.45948346, 3.81506808, 3.50156649, 10.90600741]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -6.6

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.24662572, 0.69422802, 0.18006648, 0.1175218, 0.65959162, 0.94589063]
        )
        expected_u1 = np.array(
            [0.86792097, 0.61324823, 0.83121933, 0.81063462, 0.36258741, 0.06085339]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(10)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Gumbel180
class TestGumbel180(unittest.TestCase):
    def setUp(self):
        self.copula = 4
        self.X = np.array(
            [
                [0.85, 0.75],
                [0.13, 0.3],
                [0.86, 0.82],
                [0.94, 0.88],
                [0.34, 0.34],
                [0.04, 0.06],
            ]
        )
        self.par = 4
        self.hfunc1 = np.array(
            [0.26723676, 0.95302313, 0.44018938, 0.3385766, 0.45035291, 0.7538028]
        )
        self.hfunc2 = np.array(
            [0.82844756, 0.01910045, 0.7111246, 0.8584468, 0.45035291, 0.12441571]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(4, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.73266769, 0.12856334, 0.79037844, 0.86894353, 0.29009907, 0.03742528]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [2.35483012, 0.61760498, 3.45948346, 3.81506808, 3.50156649, 10.90600741]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -6.6

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.75337428, 0.30577198, 0.81993352, 0.8824782, 0.34040838, 0.05410937]
        )
        expected_u1 = np.array(
            [0.86792097, 0.61324823, 0.83121933, 0.81063462, 0.36258741, 0.06085339]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(10)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Gumbel270
class TestGumbel270(unittest.TestCase):
    def setUp(self):
        self.copula = 5
        self.X = np.array(
            [
                [0.15, 0.75],
                [0.87, 0.3],
                [0.14, 0.82],
                [0.06, 0.88],
                [0.66, 0.34],
                [0.96, 0.06],
            ]
        )
        self.par = 4
        self.hfunc1 = np.array(
            [0.26723676, 0.95302313, 0.44018938, 0.3385766, 0.45035291, 0.7538028]
        )
        self.hfunc2 = np.array(
            [0.17155244, 0.98089955, 0.2888754, 0.1415532, 0.54964709, 0.87558429]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(4, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.01733231, 0.17143666, 0.02962156, 0.01105647, 0.04990093, 0.02257472]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [2.35483012, 0.61760498, 3.45948346, 3.81506808, 3.50156649, 10.90600741]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -6.6

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.75337428, 0.30577198, 0.81993352, 0.8824782, 0.34040838, 0.05410937]
        )
        expected_u1 = np.array(
            [0.13207903, 0.38675177, 0.16878067, 0.18936538, 0.63741259, 0.93914661]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(10)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Clayton0
class TestClayton0(unittest.TestCase):
    def setUp(self):
        self.copula = 6
        self.X = np.array(
            [
                [0.77, 0.59],
                [0.02, 0.03],
                [0.63, 0.47],
                [0.75, 0.47],
                [0.5, 0.62],
                [0.22, 0.5],
            ]
        )
        self.par = 4
        self.hfunc1 = np.array(
            [0.20525247, 0.79825506, 0.17294473, 0.08526509, 0.6805928, 0.95774959]
        )
        self.hfunc2 = np.array(
            [0.77710904, 0.10512001, 0.74837762, 0.88224315, 0.23215547, 0.01579484]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(2.5, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.56098133, 0.0191187, 0.44352899, 0.45836929, 0.46296427, 0.21810874]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [1.42164761, 21.94515913, 1.45906995, 0.82056701, 1.7064313, 0.34678816]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -3.1

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.88142578, 0.00932157, 0.71640064, 0.86060881, 0.52741148, 0.17859321]
        )
        expected_u1 = np.array(
            [0.44612896, 0.01379794, 0.48259378, 0.61447321, 0.59888367, 0.52145882]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(10)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Clayton90
class TestClayton90(unittest.TestCase):
    def setUp(self):
        self.copula = 7
        self.X = np.array(
            [
                [0.23, 0.59],
                [0.98, 0.03],
                [0.37, 0.47],
                [0.25, 0.47],
                [0.5, 0.62],
                [0.78, 0.5],
            ]
        )
        self.par = 4
        self.hfunc1 = np.array(
            [0.20525247, 0.79825506, 0.17294473, 0.08526509, 0.6805928, 0.95774959]
        )
        self.hfunc2 = np.array(
            [0.22289096, 0.89487999, 0.25162238, 0.11775685, 0.76784453, 0.98420516]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(2.5, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.02901867, 0.0108813, 0.02647101, 0.01163071, 0.15703573, 0.28189126]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [1.42164761, 21.94515913, 1.45906995, 0.82056701, 1.7064313, 0.34678816]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -3.1

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.88142578, 0.00932157, 0.71640064, 0.86060881, 0.52741148, 0.17859321]
        )
        expected_u1 = np.array(
            [0.55387104, 0.98620206, 0.51740622, 0.38552679, 0.40111633, 0.47854118]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(10)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Clayton180
class TestClayton180(unittest.TestCase):
    def setUp(self):
        self.copula = 8
        self.X = np.array(
            [
                [0.23, 0.41],
                [0.98, 0.97],
                [0.37, 0.53],
                [0.25, 0.53],
                [0.5, 0.38],
                [0.78, 0.5],
            ]
        )
        self.par = 4
        self.hfunc1 = np.array(
            [0.79474753, 0.20174494, 0.82705527, 0.91473491, 0.3194072, 0.04225041]
        )
        self.hfunc2 = np.array(
            [0.22289096, 0.89487999, 0.25162238, 0.11775685, 0.76784453, 0.98420516]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(2.5, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.20098133, 0.9691187, 0.34352899, 0.23836929, 0.34296427, 0.49810874]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [1.42164761, 21.94515913, 1.45906995, 0.82056701, 1.7064313, 0.34678816]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -3.1

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.11857422, 0.99067843, 0.28359936, 0.13939119, 0.47258852, 0.82140679]
        )
        expected_u1 = np.array(
            [0.55387104, 0.98620206, 0.51740622, 0.38552679, 0.40111633, 0.47854118]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(10)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Clayton270
class TestClayton270(unittest.TestCase):
    def setUp(self):
        self.copula = 9
        self.X = np.array(
            [
                [0.77, 0.41],
                [0.02, 0.97],
                [0.63, 0.53],
                [0.75, 0.53],
                [0.5, 0.38],
                [0.22, 0.5],
            ]
        )
        self.par = 4
        self.hfunc1 = np.array(
            [0.79474753, 0.20174494, 0.82705527, 0.91473491, 0.3194072, 0.04225041]
        )
        self.hfunc2 = np.array(
            [0.77710904, 0.10512001, 0.74837762, 0.88224315, 0.23215547, 0.01579484]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(2.5, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.20901867, 0.0008813, 0.18647101, 0.29163071, 0.03703573, 0.00189126]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [1.42164761, 21.94515913, 1.45906995, 0.82056701, 1.7064313, 0.34678816]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -3.1

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.11857422, 0.99067843, 0.28359936, 0.13939119, 0.47258852, 0.82140679]
        )
        expected_u1 = np.array(
            [0.44612896, 0.01379794, 0.48259378, 0.61447321, 0.59888367, 0.52145882]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(10)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Frank
class TestFrank(unittest.TestCase):
    def setUp(self):
        self.copula = 10
        self.X = np.array(
            [
                [0.77, 0.61],
                [0.02, 0.17],
                [0.63, 0.46],
                [0.75, 0.49],
                [0.5, 0.58],
                [0.22, 0.56],
            ]
        )
        self.par = 9
        self.hfunc1 = np.array(
            [0.19563799, 0.75148356, 0.17678686, 0.08770177, 0.67648508, 0.95575471]
        )
        self.hfunc2 = np.array(
            [0.82834015, 0.04096141, 0.82678278, 0.92057653, 0.32739298, 0.0388775]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(5.94, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.5891693, 0.0153556, 0.43923524, 0.48092144, 0.45716152, 0.21567971]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [1.46579144, 2.14630715, 1.34122348, 0.73633802, 2.026389, 0.3904649]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -0.9

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.86571877, 0.0027835, 0.68442487, 0.84104613, 0.49935103, 0.1256926]
        )
        expected_u1 = np.array(
            [0.44305797, 0.0097499, 0.44491391, 0.59525163, 0.54323227, 0.55230081]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(10)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Joe0


class TestJoe0(unittest.TestCase):
    def setUp(self):
        self.copula = 11
        self.X = np.array(
            [
                [0.14, 0.24],
                [0.82, 0.73],
                [0.13, 0.17],
                [0.05, 0.13],
                [0.67, 0.67],
                [0.96, 0.95],
            ]
        )
        self.par = 9
        self.hfunc1 = np.array(
            [0.75385539, 0.03813759, 0.57808244, 0.6225647, 0.54001594, 0.15000331]
        )
        self.hfunc2 = np.array(
            [0.22750514, 0.9774321, 0.3485888, 0.1593981, 0.54001594, 0.89408941]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(7.87, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.11887127, 0.72922851, 0.09210455, 0.03350977, 0.64358121, 0.9492954]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [2.38209096, 1.10136375, 3.21688415, 3.39539306, 6.54617948, 21.16042556]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -8.29

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.24016701, 0.72379558, 0.1751959, 0.13060127, 0.66978162, 0.95410266]
        )
        expected_u1 = np.array(
            [0.35526445, 0.82530509, 0.18223432, 0.05733888, 0.68159474, 0.9503228]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(9)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Joe90


class TestJoe90(unittest.TestCase):
    def setUp(self):
        self.copula = 12
        self.X = np.array(
            [
                [0.86, 0.24],
                [0.18, 0.73],
                [0.87, 0.17],
                [0.95, 0.13],
                [0.33, 0.67],
                [0.04, 0.95],
            ]
        )
        self.par = 9
        self.hfunc1 = np.array(
            [0.75385539, 0.03813759, 0.57808244, 0.6225647, 0.54001594, 0.15000331]
        )
        self.hfunc2 = np.array(
            [0.77249486, 0.0225679, 0.6514112, 0.8406019, 0.45998406, 0.10591059]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(7.87, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.12112873, 0.00077149, 0.07789545, 0.09649023, 0.02641879, 0.0007046]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [2.38209096, 1.10136375, 3.21688415, 3.39539306, 6.54617948, 21.16042556]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -8.29

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.24016701, 0.72379558, 0.1751959, 0.13060127, 0.66978162, 0.95410266]
        )
        expected_u1 = np.array(
            [0.64473555, 0.17469491, 0.81776568, 0.94266112, 0.31840526, 0.0496772]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(9)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Joe180


class TestJoe180(unittest.TestCase):
    def setUp(self):
        self.copula = 13
        self.X = np.array(
            [
                [0.86, 0.76],
                [0.18, 0.27],
                [0.87, 0.83],
                [0.95, 0.87],
                [0.33, 0.33],
                [0.04, 0.05],
            ]
        )
        self.par = 9
        self.hfunc1 = np.array(
            [0.24614461, 0.96186241, 0.42191756, 0.3774353, 0.45998406, 0.84999669]
        )
        self.hfunc2 = np.array(
            [0.77249486, 0.0225679, 0.6514112, 0.8406019, 0.45998406, 0.10591059]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(7.87, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.73887127, 0.17922851, 0.79210455, 0.85350977, 0.30358121, 0.0392954]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [2.38209096, 1.10136375, 3.21688415, 3.39539306, 6.54617948, 21.16042556]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -8.29

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.75983299, 0.27620442, 0.8248041, 0.86939873, 0.33021838, 0.04589734]
        )
        expected_u1 = np.array(
            [0.64473555, 0.17469491, 0.81776568, 0.94266112, 0.31840526, 0.0496772]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(9)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Joe270
class TestJoe270(unittest.TestCase):
    def setUp(self):
        self.copula = 14
        self.X = np.array(
            [
                [0.14, 0.76],
                [0.82, 0.27],
                [0.13, 0.83],
                [0.05, 0.87],
                [0.67, 0.33],
                [0.96, 0.05],
            ]
        )
        self.par = 9
        self.hfunc1 = np.array(
            [0.24614461, 0.96186241, 0.42191756, 0.3774353, 0.45998406, 0.84999669]
        )
        self.hfunc2 = np.array(
            [0.22750514, 0.9774321, 0.3485888, 0.1593981, 0.54001594, 0.89408941]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(7.87, alpha, rtol=0.05).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.02112873, 0.09077149, 0.03789545, 0.01649023, 0.02641879, 0.0107046]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [2.38209096, 1.10136375, 3.21688415, 3.39539306, 6.54617948, 21.16042556]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -8.29

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.75983299, 0.27620442, 0.8248041, 0.86939873, 0.33021838, 0.04589734]
        )
        expected_u1 = np.array(
            [0.35526445, 0.82530509, 0.18223432, 0.05733888, 0.68159474, 0.9503228]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(9)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %% Student
class TestStudent(unittest.TestCase):
    def setUp(self):
        self.copula = 15
        self.X = np.array(
            [
                [0.31, 0.46],
                [0.72, 0.83],
                [0.39, 0.46],
                [0.9, 0.91],
                [0.14, 0.07],
                [0.97, 0.92],
            ]
        )
        self.par = [0.9, 7]
        self.hfunc1 = np.array(
            [0.79590722, 0.85452116, 0.64419927, 0.67353092, 0.1040322, 0.21454224]
        )
        self.hfunc2 = np.array(
            [0.16407349, 0.24947538, 0.32142926, 0.56126804, 0.74186589, 0.94018449]
        )

    def test_random(self):
        n = 6
        np.random.seed(10)
        ur = random(self.copula, self.par, n)
        assert np.array_equal(self.X, np.around(ur, decimals=2))

    def test_fit(self):
        alpha = fit(self.copula, self.X)
        assert np.isclose(
            [0.9985818669143814, 0.8696769069009351], alpha, rtol=0.05
        ).all()

    def test_CDF(self):
        expected_cdf = np.array(
            [0.29094607, 0.70466519, 0.34884779, 0.8764998, 0.06255822, 0.91712858]
        )
        result_cdf = CDF(self.copula, self.X, self.par)
        assert np.isclose(result_cdf, expected_cdf, rtol=0.05).all()

    def test_PDF(self):
        expected_pdf = np.array(
            [1.61534923, 2.15233603, 2.27036635, 5.78829905, 3.31347292, 4.33205218]
        )
        result_pdf = PDF(self.copula, self.X, self.par)
        assert np.isclose(result_pdf, expected_pdf, rtol=0.05).all()

    def test_hfunc(self):
        result_hfunc1 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=1)
        result_hfunc2 = hfunc(self.copula, self.X[:, 0], self.X[:, 1], self.par, un=2)
        assert np.isclose(self.hfunc1, result_hfunc1, rtol=0.05).all()
        assert np.isclose(self.hfunc2, result_hfunc2, rtol=0.05).all()

    def test_hinvfunc(self):
        hinv1 = hfuncinverse(self.copula, self.X[:, 0], self.hfunc1, self.par, un=1)
        hinv2 = hfuncinverse(self.copula, self.X[:, 1], self.hfunc2, self.par, un=2)
        assert np.isclose(hinv1, self.X[:, 1], rtol=0.05).all()
        assert np.isclose(hinv2, self.X[:, 0], rtol=0.05).all()

    def test_neg_likelihood(self):
        likelihood_result = neg_likelihood(self.par, self.copula, self.X)
        assert np.around(likelihood_result, decimals=2) == -6.49

    def test_randomconditional(self):
        expected_u2 = np.array(
            [0.44631022, 0.31886433, 0.45566255, 0.92322345, 0.16314601, 0.92194905]
        )
        expected_u1 = np.array(
            [0.31859019, 0.42853726, 0.44906164, 0.92687608, 0.07697798, 0.89739553]
        )
        np.random.seed(10)
        result_u2 = randomconditional(self.copula, self.X[:, 0], self.par, 6, un=1)
        np.random.seed(11)
        result_u1 = randomconditional(self.copula, self.X[:, 1], self.par, 6, un=2)
        assert np.isclose(result_u1, expected_u1, rtol=0.05).all()
        assert np.isclose(result_u2, expected_u2, rtol=0.05).all()

    def test_bestcop(self):
        cops = list(range(1, 16))
        np.random.seed(1)
        ur = random(self.copula, self.par, 1000)
        cop, par, AIC = bestcop(cops, ur)
        assert cop == self.copula


# %%
if __name__ == "__main__":
    unittest.main()
