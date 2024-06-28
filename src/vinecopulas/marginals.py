# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:13:20 2024


"""


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import rankdata
from scipy.optimize import newton
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.linalg import cholesky
from itertools import product
import sys

# %% best fit discrete distribution


def best_fit_distributiondiscrete(data, bound=False, criterion="BIC"):
    """
    Fits the best discrete distribution to data.

    Arguments:
        *data* : The data which has to be fit as a 1-d numpy array.

        *bounds* : whether the data is bounded

        *criterion* : Either BIC, AIC or ML is used to choose the best distribution

    Returns:
     *bestdist* : the best distribution and its parameters.
    """

    # distributions
    distributions = {
        "Betabinomial": st.betabinom,
        "Binomial": st.binom,
        "Poisson": st.poisson,
        "Geometric": st.geom,
        "Hypergeometric": st.hypergeom,
        "Negative Binomial": st.nbinom,
        "Negative Hypergeometric": st.nhypergeom,
        "Zipfian": st.zipfian,
        "Log-Series": st.logser,
        "Laplacian": st.dlaplace,

    }
    if bound == True:
        distributions = {
            "Betabinomial": st.betabinom,
            "Binomial": st.binom,
            "Poisson": st.poisson,
            "Geometric": st.geom,
            "Hypergeometric": st.hypergeom,
            "Negative Binomial": st.nbinom,
        }

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for name, distribution in distributions.items():

        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if name == "Betabinomial":
                    bounds = (
                        (0, len(np.unique(data))),
                        (0, len(np.unique(data))),
                        (0, len(np.unique(data))),
                    )
                elif name == "Hypergeometric" or name == "Negative Hypergeometric":
                    bounds = (
                        (0, len((data))),
                        (0, len(np.unique(data))),
                        (0, len((data))),
                    )
                elif (
                    name == "Binomial"
                    or name == "Negative Binomial"
                    or name == "Zipfian"
                    or name == "Zipf"
                    or name == "Boltzmann"
                    or name == "Planck"
                ):
                    bounds = ((0, len(data)), (0, len(data)))
                elif (
                    name == "Poisson"
                    or name == "Geom"
                    or name == "Log-Series"
                    or name == "Laplacian"
                    or name == "Yule-Simon"
                ):
                    bounds = ((0, max(data)), (0, max(data)))
                elif name == "Fisher" or name == "Wallenius":
                    bounds = (
                        (0, len(data)),
                        (0, len(data)),
                        (0, len(data)),
                        (0, len(data)),
                    )
                elif name == "Bernoulli":
                    bounds = None
                # fit dist to data
                params = st.fit(distribution, data, bounds).params[:]

                # Separate parts of parameters

                # Calculate fitted PDF and error with fit in distribution

                n = len(data)
                k = len(params)

                log_likelihood = distribution(*params).logpmf(data).sum()

                if criterion == "BIC":
                    criterion_value = -2 * log_likelihood + np.log(n) * k
                elif criterion == "AIC":
                    criterion_value = 2 * k - 2 * log_likelihood
                elif criterion == "ML":
                    criterion_value = log_likelihood

                # identify if this distribution is better
                best_distributions.append((distribution, params, criterion_value))

        except Exception:
            pass

        bestdist = sorted(best_distributions, key=lambda x: x[2])[0]
    return bestdist


# %% best fit distribution
def best_fit_distribution(data, criterion="BIC", dists = []):
    """
    Fits the best continuous distribution to data.

    Arguments:
        *data* : The data which has to be fit as a 1-d numpy array.

        *criterion* : Either BIC, AIC or ML is used to choose the best distribution
        
        *dists* : Specify specific distributions if only specific distributions need to be tested, provided as a list.

    Returns:
     *bestdist* : the best distribution and its parameters.
    """
    # distributions
    distributions = {
        "Beta": st.beta,
        "Birnbaum-Saunders": st.burr,
        "Exponential": st.expon,
        "Extreme value": st.genextreme,
        "Gamma": st.gamma,
        "Generalized extreme value": st.genextreme,
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
    
    if len(dists)> 0:
        keys_list = list(distributions.keys())
    
        keys_list2 = []
        for i in dists:
            keys_list2.append(keys_list[i])
            
        distributions = dict((k, distributions[k]) for k in keys_list2
               if k in distributions)
    
   
        
    
    
    

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for name, distribution in distributions.items():

        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution

                n = len(data)
                k = len(params)

                log_likelihood = (
                    distribution(loc=loc, scale=scale, *arg).logpdf(data).sum()
                )

                if criterion == "BIC":
                    criterion_value = -2 * log_likelihood + np.log(n) * k
                elif criterion == "AIC":
                    criterion_value = 2 * k - 2 * log_likelihood
                elif criterion == "ML":
                    criterion_value = log_likelihood

                # identify if this distribution is better
                best_distributions.append((distribution, params, criterion_value))

        except Exception:
            pass

    return sorted(best_distributions, key=lambda x: x[2])[0]


# %% pseudodata
def pseudodata(data):
    """
    Compute the pseudo-observations for the given data (transforms data to standard uniform margins)

    Arguments:
        *data* : The data which has to be converted into pseudo data, provided as a numpy array where each column contains a separate variable (eg. x1,x2,...,xn)


    Returns:
     *u* : Pseudo data, provided as a numpy array where each column contains a separate variable (eg. u1,u2,...,un)
    """

    ranks = np.apply_along_axis(rankdata, axis=0, arr=data)
    n = data.shape[0]
    u = (ranks - 1) / (n - 1)

    return u


# %%
def pseudodiscr(xcdf, xpmf):
    """
    Compute the pseudo-observations for the given variable that is discrete.

    Arguments:
        *xcdf* : The cumulative distribution function of the variable, calculated based on the best fit discrete distribution, provided as a 1-d numpy array.

        *xpmf* : The probability mass function of the variable, calculated based on the best fit discrete distribution, provided as a 1-d numpy array.

    Returns:
     *ui* : Pseudo data of a given variable  provided as a 1-d numpy array.
    """

    # Reference: Mitskopoulos et al. 2022
    u113 = xcdf - xpmf
    ru = np.random.uniform(0, 1, len(xcdf))
    ui = u113 + ru * (xcdf - u113)
    return ui
