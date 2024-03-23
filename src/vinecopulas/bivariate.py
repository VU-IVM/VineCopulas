# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:39:03 2024


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

#%% Copulas

copulas = {1: 'Gaussian', 2 : 'Gumbel0', 3 :'Gumbel90' , 4 : 'Gumbel180', 5 : 'Gumbel270', 6 : 'Clayton0', 7 : 'Clayton90', 8 : 'Clayton180', 9: 'Clayton270', 10: 'Frank', 11: 'Joe0', 12: 'Joe90', 13: 'Joe180', 14: 'Joe270', 15: 'Student'} 

#%% fitting

def fit(cop, u):
    """
    Fits a specific copula to data.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the copulae will be fit. Column 1 contains variable u1, and column 2 contains variable u2.
     
     
    Returns:  
     *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
      
    """
    u[u==1] = 0.999999
    u[u==0] = 0.000001
    #Gaussian
    if cop == 1:
        par = np.corrcoef(st.norm.ppf(u),rowvar=False)[0][1]
        
    # Gumbel and Clayton all rotations
    if cop > 1 and cop < 10:
        res = minimize_scalar(neg_likelihood, bounds=(1, 20), args=(cop, u), method='bounded')
        par = res.x
        
    # Frank
    if cop == 10:
       res = minimize_scalar(neg_likelihood, bounds=(-20, 20), args=(cop, u), method='bounded')
       par = res.x 
       
    # Joe all rotiations
    if cop > 10 and cop < 15:
        res = minimize_scalar(neg_likelihood, bounds=(1, 20), args=(cop, u), method='bounded')
        par = res.x
     
    # Student    
    if cop == 15:
        u1 = u[:,0]
        u2 = u[:,1]
        rho = np.sin(0.5 * np.pi * st.kendalltau(u1,u2)[0])
        R = np.array([[1, rho],
                      [rho, 1]])

        # Perform Cholesky decomposition
        R= np.linalg.cholesky(R)

        def invcdf(p):
            if p <= 0.9:
                q = (1000 / 9) * p
            else:
                q = 100 * (1 - 10 * (p - 0.9))**-5
            return q

        def negloglike(mu_, u, R):
            nu_ = invcdf(mu_)
            t_values = st.t.ppf(u, nu_)
            tRinv =np.linalg.solve(R, t_values.T).T
            
            n,d = u.shape
            
            nll = -n * gammaln((nu_ + d) / 2) + n * d * gammaln((nu_ + 1) / 2) - n * (d - 1) * gammaln(nu_ / 2) \
                  + n * np.sum(np.log(np.abs(np.diag(R)))) \
                  + ((nu_ + d) / 2) * np.sum(np.log(1 + np.sum(tRinv ** 2, axis=1) / nu_)) \
                  - ((nu_ + 1) / 2) * np.sum(np.sum(np.log(1 + t_values ** 2 / nu_), axis=1), axis=0)
            
            return nll


        res =minimize_scalar(negloglike, args=(u,R), 
             bounds=(0,1), method='bounded')
        df = invcdf(res.x)
        par = [rho, df]
    return par

#%% best fit


def bestcop(cops, u):
    """
    Fits the best copula to data based on a selected list of copulas to fit to using the AIC.
    
    Arguments:
        *cops* : A list of integers referring to the copulae of interest for which the fit has to be evaluated. eg. a list of [1, 10] refers to the Gaussian and Frank copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the copulae will be fit and evaluated. Column 1 contains variable u1, and column 2 contains variable u2.
     
     
    Returns:  
     *cop* : An integer referring to the copula with the best fit. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
         
     *par* : The correlation parameters of the copula with the best fit, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
         
     *aic* : The Akaike information criterion of the copula with the best fit.
      
    """
    
    AIC = []
    PAR = []
    for cop in cops:
        par = fit(cop, u)
        if cop == 15:
            AIC.append(4 + (2 * neg_likelihood(par,cop,u)))
            PAR.append(par)
        else:
            AIC.append(2 + (2 * neg_likelihood(par,cop,u)))
            PAR.append(par)
   
    i = np.where(AIC == np.nanmin(AIC))[0][0]  
    cop = cops[i]
    par = PAR[i]
    aic = AIC[i]
    return cop, par, aic
        

#%%Copula random

def random(cop, par, n): 
    """
    Generates random numbers from a chosen copula with specific parameters.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
     
        *n* : Number of random samples to return, specified a positive integer.
     
    Returns:    
     *u* : A 2-d numpy array containing random samples with n amount of rows. Column 1 contains variable u1, and column 2 contains variable u2.
      
    """

    
    # Gaussian
    if cop == 1:
        rho = par
        rho = np.array([[1, rho], [rho,1]])
        u = st.norm.cdf(np.random.multivariate_normal(np.array([0, 0]), rho, n))
    
    if cop > 1:
        alpha = par
        
    # Gumbel 
    if cop > 1 and cop < 6:
        v1 = np.random.uniform(0.00001,0.999999,n)
        v2 = np.random.uniform(0.00001,0.999999,n)
        def equation2(w):
            return (w * (1-(np.log(w)/alpha))) - v2
        w_guess = v2
        w = newton(equation2, w_guess,maxiter=2000)
        u1 = np.exp((v1**(1/alpha)) * np.log(w))
        u2 = np.exp(((1-v1)**(1/alpha)) * np.log(w))
       
        # 90 dergees
        if cop == 3:
            u1 = 1 - u1
        # 180 dergees
        elif cop == 4:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 dergees
        elif cop == 5:
            u2 = 1 - u2
        u = np.vstack((u1,u2)).T
        
    # Clayton
    if  cop > 5 and cop < 10:
        u1 = np.random.uniform(0.00001,0.999999,n)
        v2 = np.random.uniform(0.00001,0.999999,n)
        u2 = ((u1**-alpha)*((v2**(-alpha/(1+alpha)))-1)+1)**(-1/alpha)
        # 90 degrees
        if cop == 7:
            u1 = 1 - u1
        # 180 degrees
        elif cop == 8:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 degrees
        elif cop == 9:
            u2 = 1 - u2
        u = np.vstack((u1,u2)).T
    
    # Frank
    if cop == 10:
        ui = np.random.rand(n)
        y = np.random.rand(n)
        uii = (-1/alpha)*np.log(1+((y*(1-np.exp(-alpha)))/(y*(np.exp(-alpha*ui)-1)-np.exp(-alpha*ui))))
        u = np.vstack((ui,uii)).T
    
    # Joe
    if  cop > 10 and cop < 15:
        v1 = np.random.uniform(0.00001,0.999999,n)
        v2 = np.random.uniform(0.00001,0.999999,n)
        def equation2(w):
            return w - ((1/alpha) * ((np.log((1-(1-w)**alpha))*(1-(1-w)**alpha))/((1-w)**(alpha-1)))) - v2
        w_guess = v2
        w = newton(equation2, w_guess,maxiter=2000)
        u1 = 1 - (1-(1-(1-w)**alpha)**v1)**(1/alpha)
        u2 =  1 - (1-(1-(1-w)**alpha)**(1-v1))**(1/alpha)
        # 90 degrees
        if cop == 12:
            u1 = 1 - u1
        # 180 degrees
        elif cop == 13:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 degrees
        elif cop == 14:
            u2 = 1 - u2
        u = np.vstack((u1,u2)).T
    
    
    #Student
    if cop == 15:
        alpha = par[0]
        alpha = np.array([[1.       , alpha],
               [alpha, 1.       ]])
        df = par[1]
        k = st.multivariate_t(np.array([0, 0]), shape = alpha, df = df)
        u = st.t.cdf(k.rvs(size=n), df = df)
    
    u[u <=0] = 0.000001
    u[u >=1] = 0.999999
    
  
    
    return u
        
        
#%% Copula conditional random
       
def randomconditional(cop, ui, par, n, un = 1):

    """
    Generates conditional random numbers from a chosen copula with specific parameters.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__)
        
        *ui* : A 1-d numpy array containing the samples of variable u1, if evaluated with respect to u1, or u2 if evaluated with respect to u2 on which conditional samples should be computed
        
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__)
        
        *n*: number of samples to draw 
        
        *un* : indicated with respect to which variable the conditional samples have to be drawn. if un = 1, conditional samples of u2 will be drawn based on u1, if un = 2, conditional samples of u1 will be drawn based on u2.
     
    Returns:  
        *uii* : A 1-d numpy array containing the inverse h-function of the copula evaluated with respect to u1 or u2.
      
    """

    # Gaussian
    if cop == 1:
        ui = np.full(shape = n, fill_value = ui)
        y  = np.random.uniform(0,1,n)
        x1 = st.norm.ppf(y)
        x2 = st.norm.ppf(ui)
        inner =( x1 * np.sqrt(1-par**2)) + (par * x2)
        uii = st.norm.cdf(inner)
    
    if cop > 1:
        alpha = par
        
    # Gumbel 
    if cop > 1 and cop < 6:
        # 90 degrees
        if cop == 3 and un == 1:
            ui = 1- ui
        # 180 degrees
        elif cop == 4:
            ui = 1 - ui
        # 270 degrees
        elif  cop == 5 and un == 2:
            ui = 1 - ui
        ui = np.full(shape = n, fill_value = ui)
        vi = np.random.uniform(0.00001,0.999999,n)
        w = np.exp(np.log(ui)/ (vi **(1/alpha)))
        uii = np.exp(((1-vi)**(1/alpha)) * np.log(w))
        # 90 degrees
        if cop == 3 and un == 2:
            uii = 1 - uii
        # 180 degrees
        elif cop == 4: 
            uii = 1 - uii
        # 270 degrees
        elif cop == 5  and un == 1:
            uii = 1 - uii
        
        
    # Clayton
    if  cop > 5 and cop < 10:
        # 90 degrees
        if cop == 7 and un == 1:
            ui = 1- ui
        # 180 degrees
        elif cop== 8:
            ui = 1 - ui
        # 270 degrees
        elif cop == 9 and un == 2:
            ui = 1 - ui
        ui = np.full(shape = n, fill_value = ui)
        v2 = np.random.uniform(0.00001,0.999999,n)
        uii = ((ui**-alpha)*((v2**(-alpha/(1+alpha)))-1)+1)**(-1/alpha)
        # 90 degrees
        if cop == 7 and un == 2:
            uii = 1 - uii
        # 180 degrees
        elif cop == 8: 
            uii = 1 - uii
        # 270 degrees
        elif cop == 9  and un == 1:
            uii = 1 - uii
        
  
 
    
    # Frank
    if cop == 10:
        ui = np.full(shape = n, fill_value = ui)
        p = np.random.rand(n)

        if abs(alpha) > np.log(sys.float_info.max):
            uii = (ui < 0) + np.sign(alpha) * ui  # u1 or 1-u1
        elif abs(alpha) > np.sqrt(np.finfo(float).eps):
            uii = -np.log((np.exp(-alpha * ui) * (1 - p) / p + np.exp(-alpha)) / (1 + np.exp(-alpha * ui) * (1 - p) / p)) / alpha
        else:
            uii = p
        
    
    # Joe
    if  cop > 10 and cop < 15:
        # 90
        if cop == 12 and un == 1:
            ui = 1- ui
        # 180
        elif cop == 13:
            ui = 1 - ui
        # 270
        elif cop == 14 and un == 2:
            ui = 1 - ui
        ui = np.full(shape = n, fill_value = ui)
        v1 = np.random.uniform(0.00001,0.999999,n)
        if un == 1:
            w = 1 - (1-(1-(1-ui)**(alpha))**(1/v1))**(1/alpha)
            uii =  1 - (1-(1-(1-w)**alpha)**(1-v1))**(1/alpha)
        elif un == 2:
            w =  1 - (1-(1-(1-ui)**alpha)**(1/(1-v1)))**(1/alpha)
            uii = 1 - (1-(1-(1-w)**alpha)**v1)**(1/alpha)
        # 90
        if cop == 12 and un == 2:
            uii = 1 - uii
        # 180
        elif cop == 13: 
            uii = 1 - uii
        # 270
        elif cop == 14  and un == 1:
            uii = 1 - uii
    
    
    #Student
    if cop == 15:
        ui = np.full(shape = n, fill_value = ui)
        y  = np.random.uniform(0,1,n)
        alpha = par[0]
        df = par[1]
        xi = st.t.ppf(ui, df) 
        xy= st.t.ppf(y, df) 
        inner = xy * np.sqrt(((df+xi**2)*(1-alpha**2))/(df+1)) + (alpha*xi)
        uii = st.t.cdf(inner, df = df)
    
    uii[uii <=0] = 0.000001
    uii[uii >=1] = 0.999999
    
  
    
    return uii

#%% CDF
def CDF(cop, u, par):
    
    """
    Computes the cumulative distribution function.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the CDF will be calculated. Column 1 contains variable u1, and column 2 contains variable u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
     
    Returns:  
     *p* : A 1-d numpy array containing the cumulative distribution function of the copula evaluated at u1 and u2.
      
    """
    # Gaussian
    # Reference: Schepsmeier and Stöber, 2013
    if cop == 1: 
        rho = par
        rho = np.array([[1.       , rho],
               [rho, 1.       ]])
        _, d = u.shape

        p = st.multivariate_normal.cdf(st.norm.ppf(u), mean=np.zeros(d), cov=rho)
 
    if cop > 1:
        u1 = u[:,0]
        u2 = u[:,1]
        alpha = par
    # Gumbel 0 degrees
    # Reference: Schepsmeier and Stöber, 2013
    if cop == 2:
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(u2))**alpha
        p = np.exp(-(t1+t2)**(1/alpha))
    # Gumbel 90 degrees
    elif cop == 3:
        u1 = 1 - u1
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(u2))**alpha
        p = u2 - np.exp(-(t1+t2)**(1/alpha))
    # Gumbel 180 degrees
    elif cop == 4:
        u1_u2 = u1 + u2
        u1 = 1 - u1
        u2 = 1 - u2
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(u2))**alpha
        p = u1_u2 -1 + np.exp(-(t1+t2)**(1/alpha))
    # Gumbel 270 degrees
    elif cop == 5:
        u2 = 1 - u2
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(u2))**alpha
        p = u1- np.exp(-(t1+t2)**(1/alpha))
        
    # Clayton 0 degrees  
    # Reference: Schepsmeier and Stöber, 2013
    elif cop == 6:
        p = ((u1 ** -alpha) + (u2 ** -alpha) - 1)**(-1/alpha)
    # Clayton 90 degrees
    elif cop == 7:
        u1 = 1 - u1
        p = u2 - ((u1 ** -alpha) + (u2 ** -alpha) - 1)**(-1/alpha)
    # Clayton 180 degrees
    elif cop == 8:
        u1_u2 = u1 + u2
        u2 = 1 - u2
        u1 = 1 - u1
        p =  u1_u2 -  1 + (((u1 ** -alpha) + (u2 ** -alpha) - 1)**(-1/alpha))
    # Clayton 270 degrees
    elif cop == 9:
        u2 = 1 - u2
        p = u1 - ((u1 ** -alpha) + (u2 ** -alpha) - 1)**(-1/alpha)
    
    # Frank
    # Reference: Schepsmeier and Stöber, 2013
    elif cop == 10:
        p = (-1/alpha)*np.log((1/(1-np.exp(-alpha)))*(1-np.exp(-alpha) - (1- np.exp(-alpha*u1))*(1- np.exp(-alpha*u2))))
    
    # Joe 0 degrees
    # Reference: Schepsmeier and Stöber, 2013
    elif cop == 11:
        p = 1- ((1-u1)**alpha + (1-u2)**alpha - (1-u1)**alpha *(1-u2)**alpha ) ** (1/alpha)
    # Joe 90 degrees
    elif cop == 12:
        u1 = 1 - u1
        p = u2 - (1- ((1-u1)**alpha + (1-u2)**alpha - (1-u1)**alpha *(1-u2)**alpha ) ** (1/alpha))
    # Joe 180 degrees
    elif cop == 13:
        u1_u2 = u1 + u2
        u2 = 1 - u2
        u1 = 1 - u1
        p =  u1_u2 -  1  +  1 - ((1-u1)**alpha + (1-u2)**alpha - (1-u1)**alpha *(1-u2)**alpha ) ** (1/alpha)
    # Joe 270 degrees
    elif cop == 14:
        u2 = 1 - u2
        p = u1 -  (1 - ((1-u1)**alpha + (1-u2)**alpha - (1-u1)**alpha *(1-u2)**alpha ) ** (1/alpha))

    # Student
    # Reference: Schepsmeier and Stöber, 2013
    if cop == 15:
        alpha = par[0]
        alpha = np.array([[1.       , alpha],
            [alpha, 1.       ]])
        df = par[1]
        p = st.multivariate_t.cdf(st.t.ppf(u, df), shape = alpha, df = df)
    
    #BB1
   # if cop == 16:
      #  theta = par[0]
      #  delta = par[1]
      #  p = (1 + ((((u1**-theta) - 1) **delta) + (((u2**-theta) - 1) **delta))**(1/delta))**(-1/theta)
    return p

#%% PDF
    
def PDF(cop, u, par):
    
    """
    Computes the probability density function.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the PDF will be calculated. Column 1 contains variable u1, and column 2 contains variable u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
     
    Returns:  
     *y* : A 1-d numpy array containing the probability density function of the copula evaluated at u1 and u2.
      
    """
    u[u <= 0 ] = 0.00001
    u[u >= 1 ] = 0.99999
    # Gaussian
    # Reference: Schepsmeier and Stöber, 2013
    if cop == 1: 
        rho = par
        x1 = st.norm.ppf(u[:,0])
        x2 = st.norm.ppf(u[:,1])
        rhosq = rho **2 
        y = (1/np.sqrt(1 -  rhosq )) * np.exp( -(rhosq* (x1**2 + x2**2) - (2* rho * x1 * x2))/( 2* (1-rhosq)))
 
    if cop > 1:
        u1 = u[:,0]
        u2 = u[:,1]
        alpha = par
    # Gumbel
    # Reference: Schepsmeier and Stöber, 2013
    if cop > 1 and cop < 6:    
        # 90 dergees
        if cop == 3:
            u1 = 1 - u1
        # 180 dergees
        elif cop == 4:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 dergees
        elif cop == 5:
            u2 = 1 - u2
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(u2))**alpha
        cdf =  np.exp(-(t1+t2)**(1/alpha))
        y = cdf * (1/(u1*u2)) * ((t1+t2) **(-2 + (2/alpha)))*((np.log(u1) * np.log(u2))**(alpha-1))* (1+(alpha - 1)*(t1+t2)**(-1/alpha))
    
    # Clayton
    # Reference: Schepsmeier and Stöber, 2013
    if  cop > 5 and cop < 10:
        # 90 degrees
        if cop == 7:
            u1 = 1 - u1
        # 180 degrees
        elif cop == 8:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 degrees
        elif cop == 9:
            u2 = 1 - u2
        y = ((1+alpha) * (u1 * u2) ** (-1 - alpha)) / ( ((u1 ** -alpha) + (u2 ** -alpha) - 1)**((1/alpha)  + 2))
    
    # Frank
    elif cop == 10:
        y = alpha*(-np.exp(alpha*(u1 + u2)) + np.exp(alpha*(u1 + u2 + 1)))*np.exp(alpha)/((1 - np.exp(alpha*u1))*(1 - np.exp(alpha*u2))*np.exp(alpha) + np.exp(alpha*(u1 + u2)) - np.exp(alpha*(u1 + u2 + 1)))**2

    if  cop > 10 and cop < 15:
        # 90 degrees
        if cop == 12:
            u1 = 1 - u1
        # 180 degrees
        elif cop == 13:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 degrees
        elif cop == 14:
            u2 = 1 - u2
        y = (1 - u1)**(alpha - 1)*(1 - u2)**(alpha - 1)*(-(1 - u1)**alpha*(1 - u2)**alpha + (1 - u1)**alpha + (1 - u2)**alpha)**((1 - 2*alpha)/alpha)*(alpha - (1 - u1)**alpha*(1 - u2)**alpha + (1 - u1)**alpha + (1 - u2)**alpha - 1)         

    # Student
    if cop == 15:
        n, d = u.shape
        alpha = par[0]
        alpha = np.array([[1.       , alpha],
               [alpha, 1.       ]])
        df = par[1]
        R = cholesky(alpha, lower=True)
        t = st.t.ppf(u,df)
        z = np.linalg.solve(R, t.T).T
        logSqrtDetRho = np.sum(np.log(np.diag(R)))
        const = gammaln((df + d) / 2) + (d - 1) * gammaln(df / 2) - d * gammaln((df + 1) / 2) - logSqrtDetRho
        numer = -((df + d) / 2) * np.log(1 + np.sum(z**2, axis=1) / df)
        denom = np.sum(-((df + 1) / 2) * np.log(1 + (t**2) / df), axis=1)
        y = np.exp(const + numer - denom)
        
    return y
        
#%% h function

def hfunc(cop, u1, u2, par, un = 1):
    """
    Computes the h-function (conditional CDF) of a copula with respect to variable u1 or u2.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__)
        
        *u1* : A 1-d numpy array containing the samples of variable u1
        
        *u2* : A 1-d numpy array containing the samples of variable u2
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *un* : indicated with respect to which variable the h-function has to be calculated. if un = 1, the h-function is calculated with respect to u1 (c(u2|u1)), if un = 2, the h-function is calculated with respect to u2 (c(u1|u2)).
     
    Returns:  
     *y* : A 1-d numpy array containing the h-function of the copula evaluated with respect to u1 or u2.
      
    """
    u1[u1 <=0] = 0.000001
    u2[u2 <=0] = 0.000001
    u2[u2 >=1] = 0.999999
    u1[u1 >=1] = 0.999999
    # Gaussian
    if cop == 1:
        rho = par
        if un == 1:
            x2 = st.norm.ppf(u1)
            x1 = st.norm.ppf(u2)
            inner = (x1 - (rho * x2)) / (np.sqrt(1 - rho**2))
            y = st.norm.cdf(inner)
        if un == 2:
            x1 = st.norm.ppf(u1)
            x2 = st.norm.ppf(u2)
            inner = (x1 - (rho * x2)) / (np.sqrt(1 - rho**2))
            y = st.norm.cdf(inner)
            
    if cop > 1:
        alpha = par     
        
    # Gumbel 0 degrees
    if cop == 2:
        if un == 2:
            t1 = (-np.log(u1))**alpha
            t2 = (-np.log(u2))**alpha
            y = -(np.exp(-(t1+t2)**(1/alpha)) * ((t1+t2)**((1/alpha) - 1)) * t2)/ (u2*np.log(u2))
        elif un == 1:
            t2 = (-np.log(u1))**alpha
            t1 = (-np.log(u2))**alpha
            y = -(np.exp(-(t1+t2)**(1/alpha)) * ((t1+t2)**((1/alpha) - 1)) * t2)/ (u1*np.log(u1))
    # Gumbel 90 degrees
    if cop == 3:
        t1 = (-np.log(1- u1))**alpha
        t2 = (-np.log(u2))**alpha
        if un == 1:
            y = -t1 * ((t2 + t1)**(1/alpha)) * np.exp(-((t2 + t1)**(1/alpha))) / ((1 - u1) * (t2 + t1) * np.log(1 - u1))
        elif un == 2:
            y = 1 + t2 * ((t2 + t1)**(1/alpha)) * np.exp(-((t2 + t1)**(1/alpha))) / (u2 * (t2 + t1) * np.log(u2))
    # Gumbel 180 degrees        
    if cop == 4:
        t1 = (-np.log(1- u1))**alpha
        t2 = (-np.log(1 - u2))**alpha
        if un == 1:
            y = t1 * ((t1 + t2)**(1/alpha)) * np.exp(-((t1 + t2)**(1/alpha))) / ((1 - u1) * (t1 + t2) * np.log(1 - u1)) + 1
        elif un == 2:
            y = t2 * ((t1 + t2)**(1/alpha)) * np.exp(-((t1 + t2)**(1/alpha))) / ((1 - u2) * (t1 + t2) * np.log(1 - u2)) + 1
    #Gumbel 270 degrees
    if cop == 5:
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(1-u2))**alpha
        if un == 1:
            y = 1 + t1 * ((t1 + t2)**(1/alpha)) * np.exp(-((t1 + t2)**(1/alpha))) / (u1 * (t1 + t2) * np.log(u1))
        elif un == 2:
            y = -t2 * ((t1 + t2)**(1/alpha)) * np.exp(-((t1 + t2)**(1/alpha))) / ((1 - u2) * (t1 + t2) * np.log(1 - u2))
            
    # Clayton 0 degrees 
    if cop == 6:     
        if un == 1:
            y = 1/(u1*u1**alpha*(-1 + u2**(-alpha) + u1**(-alpha))*(-1 + u2**(-alpha) + u1**(-alpha))**(1/alpha))
        if  un == 2:
            y = 1/(u2*u2**alpha*(-1 + u2**(-alpha) + u1**(-alpha))*(-1 + u2**(-alpha) + u1**(-alpha))**(1/alpha))
    #Clayton 90 degrees
    if cop == 7:
        if un == 1:
            y = 1/((1 - u1)*(1 - u1)**alpha*(-1 + (1 - u1)**(-alpha) + u2**(-alpha))*(-1 + (1 - u1)**(-alpha) + u2**(-alpha))**(1/alpha))
        elif un == 2:
            y =   1 - 1/(u2*u2**alpha*(-1 + (1 - u1)**(-alpha) + u2**(-alpha))*(-1 + (1 - u1)**(-alpha) + u2**(-alpha))**(1/alpha)) 
    #Clayton 180 degrees
    if cop == 8:
        if un == 1:
            y = 1 - 1/((1 - u1)*(1 - u1)**alpha*(-1 + (1 - u2)**(-alpha) + (1 - u1)**(-alpha))*(-1 + (1 - u2)**(-alpha) + (1 - u1)**(-alpha))**(1/alpha))
        elif un == 2:
            y = 1 - 1/((1 - u2)*(1 - u2)**alpha*(-1 + (1 - u2)**(-alpha) + (1 - u1)**(-alpha))*(-1 + (1 - u2)**(-alpha) + (1 - u1)**(-alpha))**(1/alpha))
    # Clayton 270 degrees
    if cop == 9:
        if un == 1:
            y =1 - 1/(u1*u1**alpha*(-1 + (1 - u2)**(-alpha) + u1**(-alpha))*(-1 + (1 - u2)**(-alpha) + u1**(-alpha))**(1/alpha))
          
        elif un == 2:
            y =  1/((1 - u2)*(1 - u2)**alpha*(-1 + (1 - u2)**(-alpha) + u1**(-alpha))*(-1 + (1 - u2)**(-alpha) + u1**(-alpha))**(1/alpha))

    # Frank
    if cop == 10:
        if un == 1:
            y = -(np.exp(alpha*u2) - 1)*np.exp(alpha)/(np.exp(alpha) - np.exp(alpha*(u1 + 1)) + np.exp(alpha*(u1 + u2)) - np.exp(alpha*(u2 + 1)))
        if un == 2:
            y = -(np.exp(alpha*u1) - 1)*np.exp(alpha)/(np.exp(alpha) - np.exp(alpha*(u1 + 1)) + np.exp(alpha*(u1 + u2)) - np.exp(alpha*(u2 + 1)))
            
    # Joe 0 degrees
    if cop == 11:
        if  un == 1:
            y =(1 - u1)**(alpha - 1)*(1 - (1 - u2)**alpha)*(-(1 - u1)**alpha*(1 - u2)**alpha + (1 - u1)**alpha + (1 - u2)**alpha)**((1 - alpha)/alpha)
        elif  un == 2:
            y = (1 - u2)**(alpha - 1)*(1 - (1 - u1)**alpha)*(-(1 - u1)**alpha*(1 - u2)**alpha + (1 - u1)**alpha + (1 - u2)**alpha)**((1 - alpha)/alpha)
    # Joe 90 degrees
    if cop == 12:
        if un == 1:
            y =u1**(alpha - 1)*(1 - (1 - u2)**alpha)*(-u1**alpha*(1 - u2)**alpha + u1**alpha + (1 - u2)**alpha)**((1 - alpha)/alpha)
        elif un == 2:
            y =  ((1 - u1**alpha)*(1 - u2)**alpha*(-u1**alpha*(1 - u2)**alpha + u1**alpha + (1 - u2)**alpha)**(1/alpha) + (u2 - 1)*(-u1**alpha*(1 - u2)**alpha + u1**alpha + (1 - u2)**alpha))/((u2 - 1)*(-u1**alpha*(1 - u2)**alpha + u1**alpha + (1 - u2)**alpha))
    # Joe 180 degrees
    if cop == 13:
        if un == 1:
            y = (u1*(-u1**alpha*u2**alpha + u1**alpha + u2**alpha) + u1**alpha*(u2**alpha - 1)*(-u1**alpha*u2**alpha + u1**alpha + u2**alpha)**(1/alpha))/(u1*(-u1**alpha*u2**alpha + u1**alpha + u2**alpha))
        elif un == 2:
            y = (u2*(-u1**alpha*u2**alpha + u1**alpha + u2**alpha) + u2**alpha*(u1**alpha - 1)*(-u1**alpha*u2**alpha + u1**alpha + u2**alpha)**(1/alpha))/(u2*(-u1**alpha*u2**alpha + u1**alpha + u2**alpha))
    # Joe 270 degrees
    if cop == 14:
        if un == 1:
            y =((1 - u1)**alpha*(1 - u2**alpha)*(-u2**alpha*(1 - u1)**alpha + u2**alpha + (1 - u1)**alpha)**(1/alpha) + (u1 - 1)*(-u2**alpha*(1 - u1)**alpha + u2**alpha + (1 - u1)**alpha))/((u1 - 1)*(-u2**alpha*(1 - u1)**alpha + u2**alpha + (1 - u1)**alpha))  
        elif un == 2:
            y = u2**(alpha - 1)*(1 - (1 - u1)**alpha)*(-u2**alpha*(1 - u1)**alpha + u2**alpha + (1 - u1)**alpha)**((1 - alpha)/alpha)
            
    # Student
    if cop == 15:
        alpha = par[0]
        df = par[1]
        if un == 2:
            x1 = st.t.ppf(u1, df) 
            x2 = st.t.ppf(u2, df) 
        elif un == 1:
            x1 = st.t.ppf(u2, df) 
            x2 = st.t.ppf(u1, df) 
        inner = (x1 - (alpha * x2))/ np.sqrt(((df+x2**2)*(1-alpha**2))/(df+1))
        y = st.t.cdf(inner, df = df+1)
        
    return y 

#%% h-inverse func

def hfuncinverse(cop, ui, y, par, un = 1):
    """
    Computes the inverse h-function (inverse conditional CDF) of a copula with respect to variable u1 or u2.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *ui* : A 1-d numpy array containing the samples of variable u1, if evaluated with respect to u1, or u2 if evaluated with respect to u2.
        
        *y* : A 1-d numpy array containing the h-function of the copula evaluated with respect to u1 or u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *un* : indicated with respect to which variable the h-function has to be calculated. if un = 1, the h-function is calculated with respect to u1 (c(u2|u1)), if un = 2, the h-function is calculated with respect to u2 (c(u1|u2)).
     
    Returns:  
     *uii* : A 1-d numpy array containing the inverse h-function of the copula evaluated with respect to u1 or u2.
      
    """

    #Gaussian
    if cop == 1:
        rho = par
        x1 = st.norm.ppf(y)
        x2 = st.norm.ppf(ui)
        inner =( x1 * np.sqrt(1-rho**2)) + (rho * x2)
        uii = st.norm.cdf(inner)
        
    if cop > 1:
        alpha = par
    #Gumbel
    if cop > 1 and cop < 6:
        # 0 degrees or 180 degrees
        if cop == 2 or cop == 4:
            uii_guess = ui
        # 90 degrees or 270 degrees
        else:
            uii_guess = 1- ui
        if un == 1:
            def equation(u2):
                return hfunc(cop, ui, u2, par, un) - y
            uii = newton(equation, uii_guess,maxiter=200000, tol=1e-10)
        if un == 2:
            def equation(u1):
                return hfunc(cop, u1, ui, par, un) - y
            uii = newton(equation, uii_guess,maxiter=200000, tol=1e-10)
        
        return uii
        
    # Clayton
    if cop > 5 and cop < 10:
        # 0 degrees
        v2 = y
        if cop == 6:
            uii = ((ui**-alpha)*((v2**(-alpha/(1+alpha)))-1)+1)**(-1/alpha)
        # 90 degrees
        if cop == 7:
            if un == 1:
                ui = 1- ui
                uii = ((ui**-alpha)*((v2**(-alpha/(1+alpha)))-1)+1)**(-1/alpha) 
            elif un == 2    :
                v2 = 1 - v2
                uii = 1 - ((ui**-alpha)*((v2**(-alpha/(1+alpha)))-1)+1)**(-1/alpha)
        # 180 degrees
        if cop == 8:
            ui = 1 - ui
            v2 = 1 - v2
            uii = 1- ((ui**-alpha)*((v2**(-alpha/(1+alpha)))-1)+1)**(-1/alpha) 
        # 270 degrees
        if cop == 9:
            if  un == 1 :
                v2 = 1 - v2
                uii = 1 - ((ui**-alpha)*((v2**(-alpha/(1+alpha)))-1)+1)**(-1/alpha)
            elif un == 2:
                ui = 1 - ui
                uii = ((ui**-alpha)*((v2**(-alpha/(1+alpha)))-1)+1)**(-1/alpha)
                
    # Frank
    if cop == 10:
        uii = (-1/alpha)*np.log(1+((y*(1-np.exp(-alpha)))/(y*(np.exp(-alpha*ui)-1)-np.exp(-alpha*ui))))
        
    # Joe
    if cop > 10 and cop < 15:
        # 0 degrees or 180 degrees
        if cop == 11 or cop == 13:
            uii_guess = ui
        # 90 degrees or 270 degrees
        else:
            uii_guess = 1- ui
        if un == 1:
            def equation(u2):
                return hfunc(cop, ui, u2, par, un) - y 
           
            uii = newton(equation, uii_guess,maxiter=2000, tol=1e-10)
        if un == 2:
            def equation(u1):
                return hfunc(cop, u1, ui, par, un) - y
            uii = newton(equation, uii_guess,maxiter=2000, tol=1e-10)
            
    # Student
    if cop == 15:
        alpha = par[0]
        df = par[1]
        xi = st.t.ppf(ui, df) 
        xy= st.t.ppf(y, df) 
        inner = xy * np.sqrt(((df+xi**2)*(1-alpha**2))/(df+1)) + (alpha*xi)
        uii = st.t.cdf(inner, df = df)
        
    return uii

#%% negative likelyhood
def neg_likelihood(par,cop,u):
    """
    Computes the negative likelihood function.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the CDF will be calculated. Column 1 contains variable u1, and column 2 contains variable u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
     
    Returns:  
     *l* : The negative likelihood as a scalar value.
      
    """
    l = -np.sum(np.log(PDF(cop, u, par)))
    return l

