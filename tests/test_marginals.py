# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:47:01 2024


"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.stats as st
from vinecopulas.marginals import *
import unittest


# %%
class Testdists(unittest.TestCase):
    def setUp(self):
        self.contdistributions = {
            "Beta": st.beta,
            "Birnbaum-Saunders": st.burr,
            "Exponential": st.expon,
            "Extreme value": st.genextreme,
            "Gamma": st.gamma,
            "Generalized Pareto": st.genpareto,
            "Inverse Gaussian": st.invgauss,
            "Logistic": st.logistic,
            "Log-logistic": st.fisk,
            "Lognormal": st.lognorm,
            "Nakagami": st.nakagami,
            "Normal": st.norm,
            "Rayleigh": st.rayleigh,
            "Rician": st.rice,
            "t location-scale": st.t,
            "Weibull": st.weibull_min,
        }

        self.contarguments = {
            "Beta": (2, 0.6, 0.05, 1),
            "Birnbaum-Saunders": (10.5, 4.3),
            "Exponential": (0.002, 0.97),
            "Extreme value": (-0.1, -0.04, 1),
            "Gamma": (2, 0, 1),
            "Generalized Pareto": (0.1, 0, 1),
            "Inverse Gaussian": (0.1, 0, 1),
            "Logistic": (0, 1),
            "Log-logistic": (3, 0, 1),
            "Lognormal": (0.9, 1),
            "Nakagami": (5, 0, 1),
            "Normal": (0, 1),
            "Rayleigh": (0, 1),
            "Rician": (0.9, 0, 1),
            "t location-scale": (2.7, 0, 1),
            "Weibull": (1.8, 0, 1),
        }

        self.contkeys = list(self.contdistributions.keys())

        self.discdistributions = {
            "Betabinomial": st.betabinom,
            "Binomial": st.binom,
            "Poisson": st.poisson,
            "Geometric": st.geom,
            "Hypergeometric": st.hypergeom,
            "Negative Binomial": st.nbinom,
            "Zipfian": st.zipfian,
            "Log-Series": st.logser,
            "Laplacian": st.dlaplace,
        }

        self.discarguments = {
            "Betabinomial": (5, 2.3, 0.63),
            "Binomial": (5, 0.4),
            "Poisson": (0.6, 0),
            "Geometric": (0.5, 0),
            "Hypergeometric": (20, 7, 12),
            "Negative Binomial": (5, 0.5),
            "Zipfian": (1.25, 10),
            "Log-Series": (0.6, 0),
            "Laplacian": (0.6, 0),
        }

        self.disckeys = list(self.discdistributions.keys())

    def test_best_fit_distribution(self):
        for i in self.contkeys:
            dist = self.contdistributions[i]
            args = self.contarguments[i]

            np.random.seed(10)
            u = dist.rvs(*args, size=5000)
            bestdist = best_fit_distribution(u)
            assert bestdist[0] == dist

    def test_best_fit_distributiondiscrete(self):
        for i in self.disckeys:
            dist = self.discdistributions[i]
            args = self.discarguments[i]
            np.random.seed(10)
            u = dist.rvs(*args, size=1000)
            bestdist = best_fit_distributiondiscrete(u)
            assert bestdist[0] == dist

    def test_pseudodiscr(self):
        expected_u = np.array([0.58, 0.72, 0.32, 0.42, 0.88, 0.3])
        dist = self.discdistributions[self.disckeys[0]]
        args = self.discarguments[self.disckeys[0]]
        np.random.seed(10)
        x = dist.rvs(*args, size=100)
        xcdf = dist.cdf(x, *args)  # cdf of the rings data
        xpmf = dist.pmf(x, *args)  # pmf of the rings data
        u = pseudodiscr(xcdf, xpmf)
        assert np.isclose(expected_u, u[:6], rtol=0.05).all()

    def test_pseudodata(self):
        datapath = "https://raw.githubusercontent.com/VU-IVM/vinecopula/develop/doc/sample_data.csv"  # path to data
        df = pd.read_csv(datapath)
        u = pseudodata(np.array(df)[:, :-1])
        expected_u = np.array(
            [
                [0.21, 0.19, 0.19, 0.24, 0.29, 0.27],
                [0.28, 0.16, 0.19, 0.62, 0.27, 0.27],
                [0.27, 0.25, 0.22, 0.38, 0.31, 0.31],
                [0.37, 0.28, 0.23, 0.59, 0.37, 0.33],
                [0.17, 0.11, 0.22, 0.24, 0.16, 0.25],
                [0.22, 0.21, 0.29, 0.23, 0.24, 0.29],
            ]
        )
        assert np.isclose(expected_u, u[:6, :], rtol=0.05).all()


# %%

if __name__ == "__main__":
    unittest.main()
