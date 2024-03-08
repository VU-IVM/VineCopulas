# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:13:20 2024

@author: jcl202
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

#%% best fit discrete distribution

def best_fit_distributiondiscrete(data, bound = False):
    
    """
    Fits the best discrete distribution to data.
    
    Arguments:
        *data* : The data which has to be fit as a 1-d numpy array.

        *bounds* : whether the data is bounded
     
    Returns:  
     *bestdist* : the best distribution and its parameters.
    """

    # distributions
    distributions = {
    #'Bernoulli': st.bernoulli, #not relevant unless true or false scenario
    'Betabinomial': st.betabinom,
    'Binomial': st.binom,
    #'Boltzmann': st.boltzmann,
   # 'Planck': st.planck,
    'Poisson' : st.poisson,
    'Geometric': st.geom,
    'Hypergeometric': st.hypergeom,
    'Negative Binomial': st.nbinom,
   # 'Fisher': st.nchypergeom_fisher, #slow
    #'Wallenius': st.nchypergeom_wallenius, #slow
    'Negative Hypergeometric': st.nhypergeom,
    'Zipfian': st.zipfian,
    'Log-Series': st.logser,
    'Laplacian': st.dlaplace,
   # 'Yule-Simon': st.yulesimon
   # 'Zipf': st.zipf
    }
    if bound == True:
        distributions = {
   # 'Bernoulli': st.bernoulli, #not relevant unless true or false scenario
    'Betabinomial': st.betabinom,
    'Binomial': st.binom,
    #'Boltzmann': st.boltzmann,
   # 'Planck': st.planck,
    'Poisson' : st.poisson,
    'Geometric': st.geom,
    'Hypergeometric': st.hypergeom,
    'Negative Binomial': st.nbinom,
   # 'Fisher': st.nchypergeom_fisher, #slow
  
    }
        
    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for name, distribution in distributions.items():

        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                if name == 'Betabinomial':
                    bounds = ((0,len(np.unique(data))), (0,len(np.unique(data))), (0,len(np.unique(data))))
                elif name == 'Hypergeometric' or name == 'Negative Hypergeometric':
                    bounds = ((0,len((data))), (0,len(np.unique(data))), (0,len((data))))
                elif name == 'Binomial' or name == 'Negative Binomial' or name == 'Zipfian' or name == 'Zipf' or name == 'Boltzmann' or name == 'Planck':
                    bounds = ((0,len(data)), (0,len(data)))
                elif name == 'Poisson' or name == 'Geom' or name == 'Log-Series' or name == 'Laplacian' or name == 'Yule-Simon':
                    bounds = ((0, max(data)), (0, max(data)))
                elif name ==  'Fisher' or name == 'Wallenius':
                    bounds = ((0,len(data)), (0,len(data)), (0,len(data)),  (0,len(data)))
                elif name ==  'Bernoulli':
                    bounds = None
                # fit dist to data
                params = st.fit(distribution,data,bounds).params[:]

                # Separate parts of parameters

                # Calculate fitted PDF and error with fit in distribution
                
                n = len(data)
                k = len(params)
             
                log_likelihood = distribution(*params).logpmf(data).sum()
              
                BIC = -2 * log_likelihood + np.log(n) * k

                # identify if this distribution is better
                best_distributions.append((distribution, params, BIC))
        
        except Exception:
            pass

        bestdist = sorted(best_distributions, key=lambda x:x[2])[0]
    return bestdist
#%% best fit distribution

def best_fit_distribution(data, criterion = "BIC"):
    """
    Fits the best continious distribution to data.
    
    Arguments:
        *data* : The data which has to be fit as a 1-d numpy array.

     
    Returns:  
     *bestdist* : the best distribution and its parameters.
    """
    # distributions
    distributions = {
        'Beta': st.beta,
        'Birnbaum-Saunders': st.burr,
        'Exponential': st.expon,
        'Extreme value': st.genextreme,
        'Gamma': st.gamma,
        'Generalized extreme value': st.genextreme,
        'Generalized Pareto': st.genpareto,
        'Inverse Gaussian': st.invgauss,
        'Logistic': st.logistic,
        'Log-logistic': st.fisk,
        'Lognormal': st.lognorm,
        'Nakagami': st.nakagami,
        'Normal': st.norm,
        'Rayleigh': st.rayleigh,
        'Rician': st.rice,
        't location-scale': st.t,
        'Weibull': st.weibull_min
    }


    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for name, distribution in distributions.items():

        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
            
                # Calculate fitted PDF and error with fit in distribution
                
                n = len(data)
                k = len(params)
             
                log_likelihood = distribution(loc=loc, scale=scale, *arg).logpdf(data).sum()
              
                BIC = -2 * log_likelihood + np.log(n) * k

                # identify if this distribution is better
                best_distributions.append((distribution, params, BIC))
        
        except Exception:
            pass

    
    return sorted(best_distributions, key=lambda x:x[2])[0]

#%% pseudodata
def pseudodata(data):
    
    """
    Compute the pseudo-observations for the given data (tranfers data to standard uniform margins)
    
    Arguments:
        *data* : The data which has to be converted into psuedo data, provided as a numpy array where each column contains a seperate variable (eg. x1,x2,...,xn)
        

    Returns:  
     *u* : Psuedo data, provided as a numpy array where each column contains a seperate variable (eg. u1,u2,...,un)
    """
    
    ranks = np.apply_along_axis(rankdata, axis=0, arr=data)
    n = data.shape[0]
    u = (ranks - 1) / (n - 1)
    
    return u

#%%
def pseudodiscr(xcdf, xpmf):
    
    """
    Compute the pseudo-observations for the given variable that is discrete.
    
    Arguments:
        *xcdf* : The cumulative distribution function of the variable, calculated based on the best fit discrete distribution, provided as a 1-d numpy array.
        
        *xpmf* : The probability mass function of the variable, calculated based on the best fit discrete distribution, provided as a 1-d numpy array.
     
    Returns:  
     *ui* : Psuedo data of a given variable  provided as a 1-d numpy array.
    """

    #Reference: Mitskopoulos et al. 2022 
    u113 = xcdf - xpmf
    ru = np.random.uniform(0,1,len(xcdf))
    ui = u113 + ru*(xcdf-u113)
    return ui