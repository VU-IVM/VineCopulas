# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 13:49:10 2024

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
#%% Copulas

copulas = {1: 'Gaussian', 2 : 'Gumbel0', 3 :'Gumbel90' , 4 : 'Gumbel180', 5 : 'Gumbel270', 6 : 'Clayton0', 7 : 'Clayton90', 8 : 'Clayton180', 9: 'Clayton270', 10: 'Frank', 11: 'Joe0', 12: 'Joe90', 13: 'Joe180', 14: 'Joe270', 15: 'Student'} 


#%%Copula random

def random(cop, par, n):
    """
    Generates random numbers from a chosen copula with specific parameters.
    
    Arguments:
        *cop* : An integer reffering to the copula of choice. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see...reffer to where this information would be)
     
        *n* : Number of random samples to return, specified a postive integer.
     
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
        #w[w <=0] = 0.000001
        #w[w >=1] = 0.999999
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
    Generates random numbers from a chosen copula with specific parameters.
    
    Arguments:
        
        *cop* : An integer reffering to the copula of choice. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
        
        *ui* : A 1-d numpy array containing the samples of variable u1, if evaluated with respect to u1, or u2 if evaluated with respect to u2 on which conditional samples should be computed
        
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see...reffer to where this information would be)
        
        *n*: number of samples to draw 
        
        *un* : indicated with respect to which variable the conditional samples have to ve drawn. if un = 1, onditional samples of u2 will be drawn based on u1, if un = 2, onditional samples of u1 will be drawn based on u2.
     
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
        
        *cop* : An integer reffering to the copula of choice. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
        
        *u* : A 2-d numpy array containing the samples for which the CDF will be calculated. Column 1 contains variable u1, and column 2 contains variable u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see...reffer to where this information would be)
     
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
    if cop == 16:
        theta = par[0]
        delta = par[1]
        p = (1 + ((((u1**-theta) - 1) **delta) + (((u2**-theta) - 1) **delta))**(1/delta))**(-1/theta)
    return p

#%% PDF
    
def PDF(cop, u, par):
    
    """
    Computes the probability density function.
    
    Arguments:
        
        *cop* : An integer reffering to the copula of choice. eg. a 1 reffrrs to the gaussian copula (see...reffer to where this information would be)
        
        *u* : A 2-d numpy array containing the samples for which the CDF will be calculated. Column 1 contains variable u1, and column 2 contains variable u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see...reffer to where this information would be)
     
     Returns:  
         
         *y* : A 1-d numpy array containing the cumulative distribution function of the copula evaluated at u1 and u2.
      
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
        
        *cop* : An integer reffering to the copula of choice. eg. a 1 reffrrs to the gaussian copula (see...reffer to where this information would be)
        
        *u1* : A 1-d numpy array containing the samples of variable u1
        
        *u2* : A 1-d numpy array containing the samples of variable u2
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see...reffer to where this information would be)
        
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
        
        *cop* : An integer reffering to the copula of choice. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
        
        *ui* : A 1-d numpy array containing the samples of variable u1, if evaluated with respect to u1, or u2 if evaluated with respect to u2
        
        *y* : A 1-d numpy array containing the h-function of the copula evaluated with respect to u1 or u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see...reffer to where this information would be)
        
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
        
        *cop* : An integer reffering to the copula of choice. eg. a 1 reffrrs to the gaussian copula (see...reffer to where this information would be)
        
        *u* : A 2-d numpy array containing the samples for which the CDF will be calculated. Column 1 contains variable u1, and column 2 contains variable u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see...reffer to where this information would be)
     
     Returns:  
         
         *l* : The negative likelihood as a scalar value.
      
    """
    l = -np.sum(np.log(PDF(cop, u, par)))
    return l

#%% fitting

def fit(cop, u):
    """
    Fits a copula to data.
    
    Arguments:
        
        *cop* : An integer reffering to the copula of choice. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
        
        *u* : A 2-d numpy array containing the samples for which the copulae will be fit. Column 1 contains variable u1, and column 2 contains variable u2.
     
     
     Returns:  
         
         *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see...reffer to where this information would be)
      
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
    Fits the best copula to data based on a selected list of copulas to fit to.
    
    Arguments:
        
        *cops* : A list of integers reffering to the copulae of interest for which the fit has to be evauluated. eg. a list of [1, 10] reffers to the Gaussian and Frank copula (see...reffer to where this information would be)
        
        *u* : A 2-d numpy array containing the samples for which the copulae will be fit and evaluated. Column 1 contains variable u1, and column 2 contains variable u2.
     
     
     Returns:  
         
         *cop* : An integer reffering to the copula with the best fit. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
         
         *par* : The correlation parameters of the copula with the best fit, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see...reffer to where this information would be)
         
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

    # Calculate the total number of observations
    n = data.shape[0]

    # Calculate the pseudo observations
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
    u113 = xcdf - xpmf
    ru = np.random.uniform(0,1,len(xcdf))
    ui = u113 + ru*(xcdf-u113)
    return ui
#%% fitting vinecopula
def vinecop(u1, copsi):
    
    """
    Fit a regular vine copula to data.
    
    Arguments:
        
        *u1* :  the data, provided as a numpy array where each column contains a seperate variable (eg. u1,u2,...,un), which have already been transferred to standard uniform margins (0<= u <= 1)
        *copsi* : A list of integers reffering to the copulae of interest for which the fit has to be evauluated in the vine copula. eg. a list of [1, 10] reffers to the Gaussian and Frank copula (see...reffer to where this information would be)
     
     Returns:  
         
         *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer reffers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.
         
         *p* : Parameters of the bivariate copulae provided as a triangular matrix.
         
         *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers reffering to the copulae with the best fit. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)

    """
    v1 = []
    v2 = []
    tauabs = []
    dimen = u1.shape[1]
    for i in range(dimen-1):
        for j in range(i+1,dimen):
            v1.append(int(i))
            v2.append(int(j))
            tauabs.append(abs(st.kendalltau(u1[:,i], u1[:,j])[0]))
        
       
    order1 = pd.DataFrame({'v1': v1, 'v2': v2, 'tauabs': tauabs}  ) 
    order1 = order1.sort_values(by='tauabs', ascending=False).reset_index(drop=True)

    inde = []
    for i in range(len(order1)):
        if i == 0:
            order2 = order1.head(1)
        else:
            if (order1.v1[i] in list(order1.v2[:i]) or order1.v1[i] in list(order1.v1[:i]) ) and ( order1.v2[i] in list(order1.v2[:i]) or order1.v2[i] in list(order1.v1[:i])):
                continue
            else:
                inde.append(i)
                order2 = pd.concat([order2, order1.loc[i].to_frame().T], ignore_index=True)
    if len(order2) < (dimen-1):
        for i in range(1, len(order1)):
            if i in inde:
                continue
            lst = list(order2.v2[order2.v1 == order1.v2[i]]) + list(order2.v1[order2.v2 == order1.v2[i]]) +list(order2.v2[order2.v1 == order1.v1[i]]) + list(order2.v1[order2.v2 == order1.v1[i]]) 
            if len(lst) == len(set(lst)):
                order2 = pd.concat([order2, order1.loc[i].to_frame().T], ignore_index=True)
            if len(order2) == dimen-1:
                break

            
    order1 = order2
    del order2
    rhos = []
    node = []
    cops = []
    v1_1 = []
    v2_1 =  []
    aics = []
    for i in range(len(order1)):
        v1i = int(order1.v1[i])
        v2i = int(order1.v2[i])
        u3=np.vstack((u1[:,v1i] , u1[:,v2i])).T
        cop, rho, aic = bestcop(copsi, u3)
        aics.append(aic)
        rhos.append(rho)
        cops.append(cop)
        node.append([v1i , v2i])
        v1_1.append(u1[:,v1i])
        v2_1.append(u1[:,v2i])
    v1_1 = np.array(v1_1).T #v1
    v2_1 = np.array(v2_1).T #v2
        
    order1['rhos'] = rhos
    order1['node'] = node
    order1['tree'] = 0
    order1['cop'] = cops
    order1['AIC'] = aics

    v1 = []
    v2 = []
    ktau = []
    rhos = []
    node = []
    cops = []
    v1_k = []
    v2_k = []
    aics = []
    for i in range(len(order1)):
        v1i = int(order1.v1[i])
        v2i = int(order1.v2[i])
        copi = int(order1.cop[i])
        pari =  order1.rhos[i]
        if i == (len(order1) - 1):
                 continue
        for j in np.where(np.array([item == v1i for item in list(order1.v1[i+1:])]) | np.array([item == v1i for item in list(order1.v2[i+1:])])| np.array([item == v2i for item in list(order1.v1[i+1:])])| np.array([item == v2i for item in list(order1.v2[i+1:])]))[0] +i + 1:
            v1j = int(order1.v1[j])
            v2j = int(order1.v2[j])
            copj = int(order1.cop[j])
            parj =  order1.rhos[j]
            v1.append(order1.node[i])
            v2.append(order1.node[j])
            lst = order1.node[i] + order1.node[j]
            s= max(set(lst), key=lst.count)
            ui1 = v1_1[:,i]
            ui2 = v2_1[:,i]
            uj1 = v1_1[:,j]
            uj2 = v2_1[:,j]
            if v1i == s:
                uni = 1
                vi1 = v2i
            else:
                uni = 2
                vi1 = v1i
            if v1j == s:
                unj = 1
                vj1 = v2j
            else:
                unj = 2
                vj1 = v1j
                
            v1igs = hfunc(copi, ui1, ui2, pari, un = uni) 
            v2jgs = hfunc(copj, uj1, uj2, parj, un = unj)
            ktau.append(abs(st.kendalltau(v1igs ,  v2jgs  )[0]))
           # rhos.append(GaussianCopulafit(np.vstack((v1igs ,  v2jgs )).T))
            v1_k.append(v1igs)
            v2_k.append(v2jgs)
            node.append([vi1, vj1, 'g', s])
            
      
    k = 2

    orderk =  pd.DataFrame({'v1': v1, 'v2': v2, 'tauabs': ktau, 'node': node}  )

    if len(orderk) > dimen-k:
        orderk = orderk.sort_values(by='tauabs', ascending=False)
        indexes = list(orderk.index)
        orderk = orderk.reset_index(drop=True)
        inde = []
        inde2 = []
        for i in range(len(orderk)):
            if i == 0:
                order = orderk.head(1)
                inde.append(indexes[i])
                
            else:
                if (orderk.v1[i] in list(orderk.v2[:i]) or orderk.v1[i] in list(orderk.v1[:i]) ) and ( orderk.v2[i] in list(orderk.v2[:i]) or orderk.v2[i] in list(orderk.v1[:i])):
                    continue
                else:
                    inde.append(indexes[i])
                    inde2.append(i)
                    order = pd.concat([order, orderk.loc[i].to_frame().T], ignore_index=True)

        if len(order) < (dimen-k):
            for i in range(1, len(orderk)):
                if i in inde2:
                    continue
                lst = list(order.v2.astype(str)[order.v1.astype(str) == str(orderk.v1[i])]) + list(order.v1.astype(str)[order.v2.astype(str) == str(orderk.v1[i])])+ list(order.v2.astype(str)[order.v1.astype(str) == str(orderk.v2[i])]) + list(order.v1.astype(str)[order.v2.astype(str) == str(orderk.v2[i])])
                if len(lst) == len(set(lst)):
                    inde.append(indexes[i])
                    order = pd.concat([order, orderk.loc[i].to_frame().T], ignore_index=True)
                if len(order) == dimen-k:
                    break
        orderk = order
        orderk = orderk.sort_values(by='tauabs', ascending=False)
        
        v1_k = np.array([v1_k[ind] for ind in inde]).T
        v2_k = np.array([v2_k[ind] for ind in inde]).T
        orderk = orderk.reset_index(drop=True)
        orderk['tree'] = k - 1
        v1_2 = v1_k.copy()
        v2_2 = v2_k.copy()
        order2 = orderk.copy()
        


    else:
        orderk = orderk.sort_values(by='tauabs', ascending=False)
       
        v1_k = np.array([v1_k[ind] for ind in orderk.index]).T
        v2_k = np.array([v2_k[ind] for ind in orderk.index]).T
        orderk = orderk.reset_index(drop=True)
        orderk['tree'] = k - 1
        v1_2 = v1_k.copy()
        v2_2 = v2_k.copy()
        order2 = orderk.copy()
        
    for i in range(len(order2)):
        u3=np.vstack((v1_2[:,i] , v2_2[:,i])).T
        cop, rho, aic= bestcop(copsi, u3)
        aics.append(aic)
        rhos.append(rho)
        cops.append(cop)
        
    order2['rhos'] = rhos
    order2['cop'] = cops
    order2['AIC'] = aics
        

    if dimen > 3:
        for k in range(3,dimen):
            order = locals()['order' + str(k-1)].copy()
            v1s =  locals()['v1_' + str(k-1)].copy()
            v2s =  locals()['v2_' + str(k-1)].copy()
            v1_k = []
            v2_k = []
            v1 = []
            v2 = []
            ktau = []
            rhos = []
            cops = []
            node = []
            lk = []
            aics = []
            rk = []
            for i in range(len(order)):
                v1i = order.v1[i].copy()
                v2i = order.v2[i].copy()
                copi = int(order.cop[i])
                pari =  order.rhos[i]
                if i == (len(order) - 1):
                         continue
                for j in np.where(np.array([item == v1i for item in list(order.v1[i+1:])]) | np.array([item == v1i for item in list(order.v2[i+1:])])| np.array([item == v2i for item in list(order.v1[i+1:])])| np.array([item == v2i for item in list(order.v2[i+1:])]))[0] +i + 1:
                    v1i = order.v1[i].copy()
                    v2i = order.v2[i].copy()
                    copj = int(order.cop[j])
                    parj =  order.rhos[j]
                    nodei = order.node[i]
                    nodej =  order.node[j]
                    v1j =  order.v1[j].copy()
                    v2j =  order.v2[j].copy()
                    v1.append(nodei)
                    v2.append(nodej)
                    n = 2

                    ri = nodei[n+1:]
                    rj = nodej[n+1:]
                
                    if 'g' in v1j:
                        v1j.remove('g')
                        v2j.remove('g')
                        v1i.remove('g')
                        v2i.remove('g')
                        
                        
                        
                    
                    if rj == ri:
                        if len(v1j) == 2:
                            lst = nodei[:n]  +  nodej[:n] 
                            r3 = [max(set(lst), key=lst.count)]
                            li = [value for value in nodei[:n] if value not in r3]
                            lj = [value for value in nodej[:n] if value not in r3]
                            r = list(np.unique(ri + r3))
                        else:
                            lst = v1i[:n] +  v2i[:n] + v1j[:n] +  v2j[:n]
                            for s in ri:
                                lst = [x for x in lst if x != s]
                            r3 = [max(set(lst), key=lst.count)]
                            li = [value for value in nodei[:n] if value not in r3]
                            lj = [value for value in nodej[:n] if value not in r3]
                            r = list(np.unique(ri + r3))
                        l = list(np.unique(li + lj))
                        #node.append(li + lj + ['g'] + ri + r3)
                    else:
                        r = list(np.unique(ri + rj))
                        li = [value for value in nodei[:n] if value not in rj]
                        lj = [value for value in nodej[:n] if value not in ri]
                        if li == lj:
                            lst = v1i[:n] +  v2i[:n] + v1j[:n] +  v2j[:n] 
                            lst = [x for x in lst if x != li[0]]
                            l3 = [min(set(lst), key=lst.count)]
                            l  =  list(np.unique(li + l3))
                        else:
                            l = list(np.unique(li + lj))
                    ui1 = v1s[:,i]
                    ui2 = v2s[:,i]
                    uj1 = v1s[:,j]
                    uj2 = v2s[:,j]
                    if set(r).issubset(set(v1i)):
                        uni = 1
                    elif set(r).issubset(set(v2i)):
                        uni = 2
                    elif set(rj).issubset(set(v2i[1:])):
                        uni= 2
                    elif set(rj).issubset(set(v1i[1:])):
                        uni = 1
                    if set(r).issubset(set(v1j)):
                        unj = 1
                    elif set(r).issubset(set(v2j)):
                        unj = 2
                    elif set(ri).issubset(set(v2j[1:])):
                        unj = 2
                    elif set(ri).issubset(set(v1j[1:])):
                        unj = 1
                        
                    v1igs = hfunc(copi, ui1, ui2, pari, un = uni) 
                    v2jgs = hfunc(copj, uj1, uj2, parj, un = unj)
                    ktau.append(abs(st.kendalltau(v1igs ,  v2jgs  )[0]))
                   # rhos.append(GaussianCopulafit(np.vstack((v1igs ,  v2jgs )).T))
                    v1_k.append(v1igs)
                    v2_k.append(v2jgs)
                       # node.append(li + lj + ['g'] + ri + rj)
                       
                   
                    del uj1,ui1,ui2,uj2
                    
                    node.append(l + ['g'] + r)
                    lk.append(l)
                    rk.append(r)
            orderk =  pd.DataFrame({'v1': v1, 'v2': v2, 'tauabs': ktau, 'node': node, 'l': lk, 'r': rk}  )
            
            if len(orderk) > dimen-k:
                orderk = orderk.sort_values(by='tauabs', ascending=False)
                indexes = list(orderk.index)
                orderk = orderk.reset_index(drop=True)
                inde = []
                inde2 = []
                for i in range(len(orderk)):
                    if i == 0:
                        order = orderk.head(1)
                        inde.append(indexes[i])
                        l = orderk.l[i]
                        
                    else:
                        if ((orderk.v1[i] in list(orderk.v2[:i]) or orderk.v1[i] in list(orderk.v1[:i]) ) and ( orderk.v2[i] in list(orderk.v2[:i]) or orderk.v2[i] in list(orderk.v1[:i]))):
                            continue
                        else:
                            inde.append(indexes[i])
                            inde2.append(i)
                            order = pd.concat([order, orderk.loc[i].to_frame().T], ignore_index=True)
                            l = l + orderk.l[i]
                if len(order) < (dimen-k):
                    for i in range(1, len(orderk)):
                        if i in inde2:
                            continue
                        lst = list(order.v2.astype(str)[order.v1.astype(str) == str(orderk.v1[i])]) + list(order.v1.astype(str)[order.v2.astype(str) == str(orderk.v1[i])])+ list(order.v2.astype(str)[order.v1.astype(str) == str(orderk.v2[i])]) + list(order.v1.astype(str)[order.v2.astype(str) == str(orderk.v2[i])])
                        if len(lst) == len(set(lst)):
                            order = pd.concat([order, orderk.loc[i].to_frame().T], ignore_index=True)
                            inde.append(indexes[i])
                        if len(order) == dimen-k:
                            break
                orderk = order
                orderk = orderk.sort_values(by='tauabs', ascending=False)
                
                v1_k = np.array([v1_k[ind] for ind in inde]).T
                v2_k = np.array([v2_k[ind] for ind in inde]).T
                orderk = orderk.reset_index(drop=True)
                orderk['tree'] = k - 1
                for j in range(len(orderk)):
                    u3=np.vstack((v1_k[:,j] , v2_k[:,j])).T
                    cop, rho , aic = bestcop(copsi, u3)
                    aics.append(aic)
                    rhos.append(rho)
                    cops.append(cop)
                    
                orderk['rhos'] = rhos
                orderk['cop'] = cops
                orderk['AIC'] = aics
                locals()['v1_' + str(k)] = v1_k
                locals()['v2_' + str(k)] = v2_k
                locals()['order' + str(k)] = orderk
                
            

            
            else:
                orderk = orderk.sort_values(by='tauabs', ascending=False)
               
                v1_k = np.array([v1_k[ind] for ind in orderk.index]).T
                v2_k = np.array([v2_k[ind] for ind in orderk.index]).T
                orderk = orderk.reset_index(drop=True)
                orderk['tree'] = k - 1
                    
                for j in range(len(orderk)):
                    u3=np.vstack((v1_k[:,j] , v2_k[:,j])).T
                    cop, rho, aic = bestcop(copsi, u3)
                    aics.append(aic)
                    rhos.append(rho)
                    cops.append(cop)
                    
                orderk['rhos'] = rhos
                orderk['cop'] = cops
                orderk['AIC'] = aics
                locals()['v1_' + str(k)] = v1_k
                locals()['v2_' + str(k)] = v2_k
                locals()['order' + str(k)] = orderk
            


    order = pd.DataFrame(columns  = order1.columns)
    for i in range(1,dimen):
        order = pd.concat([order, locals()['order' + str(i)]]).reset_index(drop=True)

        

    a = np.empty((dimen,dimen))
    c = np.empty((dimen,dimen))
    a[:] = np.nan
    c[:] = np.nan
   


    order['used'] = 0
    combinations = list(product([True, False], repeat=dimen))
    for i in list(range(dimen-1))[::-1]:
        k1 = sorted(np.array(order[(order.tree == i) & (order['used'] == 0)].node.iloc[0][:2]).astype(int))[::-1]
        order.loc[(order['tree'] == i) & (order['used'] == 0), 'used'] = 1
        t1 = i - 1
        ii = dimen-2-i
        a[i:dimen-ii,ii] = k1
        s = k1[-1]
        for j in list(range(0,i))[::-1]:
            orde = order[(order.tree == j) & (order['used'] == 0)]
            for k in range(len(orde)):
                arr = np.array(orde.node.iloc[k][:2]).astype(int)
                if np.isin(s,arr) == True:
                    inde = orde.iloc[k].name
                    a[j, ii] = arr[arr!=s][0]
                    order['used'][inde] = 1

    a[0,dimen-1] = a[0,dimen-2] 
    orderk = pd.DataFrame(columns  = order.columns)
    p = np.empty((dimen,dimen))
    p[:] = np.nan
    p = p.astype(object)
    for i in list(range(dimen-1)):
        orde = order[order.tree == i]
        for k in list(range(dimen-1-i)):
            ak = a[:,k]
            akn = np.array([ak[-1-k], ak[i]]).astype(int)
            for j in range(len(orde)):
                arr = np.array(orde.node.iloc[j][:2]).astype(int)
                if sum(np.isin(akn,arr)) == 2:
                    orderj = order.loc[[orde.index[j]]]
                    p[i,k] = orderj.rhos.iloc[0]
                    c[i,k] =  orderj.cop.iloc[0]
                    if i == 0:
                        orderj.node.iloc[0] = list(akn)
                    else:
                        orderj.node.iloc[0] = list(akn)  + ['|'] + list((ak.astype(int)[:i])[::-1])
                    orderk = pd.concat([orderk, orderj]).reset_index(drop=True)
                    
                    

    for i in list(range(dimen-1)):
        orde = orderk[orderk.tree == i].reset_index(drop=True)
        print('** Tree: ', i)
        for j in range(len(orde)):
            print(orde.node[j], copulas[int(orde.cop[j])], ' ---> parameters = ', orde.rhos[j])
    return a, p, c 
#%% fitting vine copula with condition

def copconditional(u1, vint, copsi):
    """
    Fit a regular vine copula which allows for a conditional sample of a variable of interest.
    
    Arguments:
        
        *u1* :  the data, provided as a numpy array where each column contains a seperate variable (eg. u1,u2,...,un), which have already been transferred to standard uniform margins (0<= u <= 1)
        *vint* : the variable of interest, provided as an integere that reffers to the variables column number in u1, where the first column is 0 and the second column is 1, etc.
        *copsi* : A list of integers reffering to the copulae of interest for which the fit has to be evauluated in the vine copula. eg. a list of [1, 10] reffers to the Gaussian and Frank copula (see...reffer to where this information would be)
     
     Returns:  
         
         *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer reffers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.
         
         *p* : Parameters of the bivariate copulae provided as a triangular matrix.
         
         *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers reffering to the copulae with the best fit. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)

    """
    v1 = []
    v2 = []
    tauabs = []
    dimen = u1.shape[1]
    for i in range(dimen-1):
        for j in range(i+1,dimen):
            v1.append(int(i))
            v2.append(int(j))
            tauabs.append(abs(st.kendalltau(u1[:,i], u1[:,j])[0]))
        
       
    order1 = pd.DataFrame({'v1': v1, 'v2': v2, 'tauabs': tauabs}  ) 
    order1 = order1.sort_values(by='tauabs', ascending=False).reset_index(drop=True)

    inde = []
    for i in range(len(order1)):
        if i == 0:
            order2 = order1.head(1)
        else:
            if (order1.v1[i] in list(order1.v2[:i]) or order1.v1[i] in list(order1.v1[:i]) ) and ( order1.v2[i] in list(order1.v2[:i]) or order1.v2[i] in list(order1.v1[:i])):
                continue
            if (vint == order1.v1[i] or vint == order1.v2[i] ) and (vint in list(order1.v2[:i]) or vint in list(order1.v1[:i]) ) :
                continue
            else:
                inde.append(i)
                order2 = pd.concat([order2, order1.loc[i].to_frame().T], ignore_index=True)
    if len(order2) < (dimen-1):
        for i in range(1, len(order1)):
            if i in inde:
                continue
            if (vint == order1.v1[i] or vint == order1.v2[i] ) and (vint in list(order1.v2[:i]) or vint in list(order1.v1[:i]) ) :
                continue
            lst = list(order2.v2[order2.v1 == order1.v2[i]]) + list(order2.v1[order2.v2 == order1.v2[i]]) +list(order2.v2[order2.v1 == order1.v1[i]]) + list(order2.v1[order2.v2 == order1.v1[i]]) 
            if len(lst) == len(set(lst)):
         
                order2 = pd.concat([order2, order1.loc[i].to_frame().T], ignore_index=True)
            if len(order2) == dimen-1:
                break

            
    order1 = order2
    del order2
    rhos = []
    node = []
    v1_1 = []
    v2_1 =  []
    cops = []
    aics = []
    for i in range(len(order1)):
        v1i = int(order1.v1[i])
        v2i = int(order1.v2[i])
        u3=np.vstack((u1[:,v1i] , u1[:,v2i])).T
        cop, rho, aic = bestcop(copsi, u3)
        aics.append(aic)
        rhos.append(rho)
        cops.append(cop)
        node.append([v1i , v2i])
        v1_1.append(u1[:,v1i])
        v2_1.append(u1[:,v2i])
    v1_1 = np.array(v1_1).T
    v2_1 = np.array(v2_1).T
        
    order1['rhos'] = rhos
    order1['node'] = node
    order1['tree'] = 0
    order1['cop'] = cops
    order1['AIC'] = aics

    v1 = []
    v2 = []
    ktau = []
    rhos = []
    cops = []
    node = []
    aics = []
    v1_k = []
    v2_k = []
    l = []
    r = []
    for i in range(len(order1)):
        v1i = int(order1.v1[i])
        v2i = int(order1.v2[i])
        copi = int(order1.cop[i])
        pari =  order1.rhos[i]
        if i == (len(order1) - 1):
                 continue
        for j in np.where(np.array([item == v1i for item in list(order1.v1[i+1:])]) | np.array([item == v1i for item in list(order1.v2[i+1:])])| np.array([item == v2i for item in list(order1.v1[i+1:])])| np.array([item == v2i for item in list(order1.v2[i+1:])]))[0] +i + 1:
            v1j = int(order1.v1[j])
            v2j = int(order1.v2[j])
            copj = int(order1.cop[j])
            parj =  order1.rhos[j]
            v1.append(order1.node[i])
            v2.append(order1.node[j])
            lst = order1.node[i] + order1.node[j]
            s= max(set(lst), key=lst.count)
            ui1 = v1_1[:,i]
            ui2 = v2_1[:,i]
            uj1 = v1_1[:,j]
            uj2 = v2_1[:,j]
            if v1i == s:
                uni = 1
                vi1 = v2i
            else:
                uni = 2
                vi1 = v1i
            if v1j == s:
                unj = 1
                vj1 = v2j
            else:
                unj = 2
                vj1 = v1j
                
            v1igs = hfunc(copi, ui1, ui2, pari, un = uni) 
            v2jgs = hfunc(copj, uj1, uj2, parj, un = unj)
            ktau.append(abs(st.kendalltau(v1igs ,  v2jgs  )[0]))
           # rhos.append(GaussianCopulafit(np.vstack((v1igs ,  v2jgs )).T))
            v1_k.append(v1igs)
            v2_k.append(v2jgs)
            node.append([vi1, vj1, 'g', s])
            l.append([vi1, vj1])
            r.append([s])
            
      
    k = 2

    orderk =  pd.DataFrame({'v1': v1, 'v2': v2, 'tauabs': ktau, 'node': node, 'l': l, 'r': r}  )

    if len(orderk) > dimen-k:
        orderk = orderk.sort_values(by='tauabs', ascending=False)
        orderk = orderk.reset_index(drop=True)
        indexes = list(orderk.index)
        inde = []
        inde2 = []
        for i in range(len(orderk)):
            if i == 0:
                order = orderk.head(1)
                inde.append(indexes[i])
                l = orderk.l[i]
                
            else:
                if ((vint in l) and (vint in orderk.l[i])) or ((orderk.v1[i] in list(orderk.v2[:i]) or orderk.v1[i] in list(orderk.v1[:i]) ) and ( orderk.v2[i] in list(orderk.v2[:i]) or orderk.v2[i] in list(orderk.v1[:i]))):
                    continue
                else:
                    inde.append(indexes[i])
                    inde2.append(i)
                    order = pd.concat([order, orderk.loc[i].to_frame().T], ignore_index=True)
                    l = l + orderk.l[i]
        Tf = order.apply(lambda row: ((order['v1'].apply(lambda x: set(x) == set(row['v1']))).sum() > 1) or
                                                    ((order['v2'].apply(lambda x: set(x) == set(row['v2']))).sum() > 1), axis=1)
        if sum(Tf) < len(Tf):
            for q in np.where(Tf == False)[0]:
                for i in range(1, len(orderk)):
                    if i in inde:
                        continue
                    if ((vint in l) and (vint in orderk.l[i])):
                        continue
                    if (orderk.v2[i] == order.v1[q]) or (orderk.v1[i] == order.v1[q]) or (orderk.v1[i] == order.v2[q]) or (orderk.v1[i] == order.v2[q]):

                        order = pd.concat([order, orderk.loc[i].to_frame().T], ignore_index=True)
                        inde.append(indexes[i])
                        break
                         
        if len(order) < (dimen-k):
            for i in range(1, len(orderk)):
                if i in inde:
                    continue
                if ((vint in l) and (vint in orderk.l[i])):
                    continue
                lst = list(order.v2.astype(str)[order.v1.astype(str) == str(orderk.v1[i])]) + list(order.v1.astype(str)[order.v2.astype(str) == str(orderk.v1[i])])+ list(order.v2.astype(str)[order.v1.astype(str) == str(orderk.v2[i])]) + list(order.v1.astype(str)[order.v2.astype(str) == str(orderk.v2[i])])
                if len(lst) == len(set(lst)):
                    order = pd.concat([order, orderk.loc[i].to_frame().T], ignore_index=True)
                    inde.append(indexes[i])
                if len(order) == dimen-k:
                    break
        orderk = order
        orderk = orderk.sort_values(by='tauabs', ascending=False)
        
        v1_k = np.array([v1_k[ind] for ind in inde]).T
        v2_k = np.array([v2_k[ind] for ind in inde]).T
        orderk = orderk.reset_index(drop=True)
        orderk['tree'] = k - 1
        v1_2 = v1_k.copy()
        v2_2 = v2_k.copy()
        order2 = orderk.copy()




    else:
        orderk = orderk.sort_values(by='tauabs', ascending=False)
       
        v1_k = np.array([v1_k[ind] for ind in orderk.index]).T
        v2_k = np.array([v2_k[ind] for ind in orderk.index]).T
        orderk = orderk.reset_index(drop=True)
        orderk['tree'] = k - 1
        v1_2 = v1_k.copy()
        v2_2 = v2_k.copy()
        order2 = orderk.copy()
        
        
    for i in range(len(order2)):
        u3=np.vstack((v1_2[:,i] , v2_2[:,i])).T
        cop, rho, aic = bestcop(copsi, u3)
        aics.append(aic)
        rhos.append(rho)
        cops.append(cop)
        
    order2['rhos'] = rhos
    order2['cop'] = cops
    order2['AIC'] = aics
    
    if dimen > 3:
        for k in range(3,dimen):
            order = locals()['order' + str(k-1)].copy()
            v1s =  locals()['v1_' + str(k-1)].copy()
            v2s =  locals()['v2_' + str(k-1)].copy()
            v1_k = []
            v2_k = []
            v1 = []
            v2 = []
            cops = []
            ktau = []
            rhos = []
            node = []
            lk = []
            rk = []
            aics = []
            for i in range(len(order)):
                v1i = order.v1[i].copy()
                v2i = order.v2[i].copy()
                copi = int(order1.cop[i])
                pari =  order1.rhos[i]
                if i == (len(order) - 1):
                         continue
                for j in np.where(np.array([item == v1i for item in list(order.v1[i+1:])]) | np.array([item == v1i for item in list(order.v2[i+1:])])| np.array([item == v2i for item in list(order.v1[i+1:])])| np.array([item == v2i for item in list(order.v2[i+1:])]))[0] +i + 1:
                    v1i = order.v1[i].copy()
                    v2i = order.v2[i].copy()
                    copj = int(order1.cop[j])
                    parj =  order1.rhos[j]
                    nodei = order.node[i]
                    nodej =  order.node[j]
                    v1j =  order.v1[j].copy()
                    v2j =  order.v2[j].copy()
                    v1.append(nodei)
                    v2.append(nodej)
                    n = 2

                    ri = nodei[n+1:]
                    rj = nodej[n+1:]
                
                    if 'g' in v1j:
                        v1j.remove('g')
                        v2j.remove('g')
                        v1i.remove('g')
                        v2i.remove('g')
                        
                        
                        
                    
                    if rj == ri:
                        if len(v1j) == 2:
                            lst = nodei[:n]  +  nodej[:n] 
                            r3 = [max(set(lst), key=lst.count)]
                            li = [value for value in nodei[:n] if value not in r3]
                            lj = [value for value in nodej[:n] if value not in r3]
                            r = list(np.unique(ri + r3))
                        else:
                            lst = v1i[:n] +  v2i[:n] + v1j[:n] +  v2j[:n]
                            for s in ri:
                                lst = [x for x in lst if x != s]
                            r3 = [max(set(lst), key=lst.count)]
                            li = [value for value in nodei[:n] if value not in r3]
                            lj = [value for value in nodej[:n] if value not in r3]
                            r = list(np.unique(ri + r3))
                        l = list(np.unique(li + lj))
                        #node.append(li + lj + ['g'] + ri + r3)
                    else:
                        r = list(np.unique(ri + rj))
                        li = [value for value in nodei[:n] if value not in rj]
                        lj = [value for value in nodej[:n] if value not in ri]
                        if li == lj:
                            lst = v1i[:n] +  v2i[:n] + v1j[:n] +  v2j[:n] 
                            lst = [x for x in lst if x != li[0]]
                            l3 = [min(set(lst), key=lst.count)]
                            l  =  list(np.unique(li + l3))
                        else:
                            l = list(np.unique(li + lj))
                    ui1 = v1s[:,i]
                    ui2 = v2s[:,i]
                    uj1 = v1s[:,j]
                    uj2 = v2s[:,j]
                    if set(r).issubset(set(v1i)):
                        uni = 1
                    elif set(r).issubset(set(v2i)):
                        uni = 2
                    elif set(rj).issubset(set(v2i[1:])):
                        uni= 2
                    elif set(rj).issubset(set(v1i[1:])):
                        uni = 1
                    if set(r).issubset(set(v1j)):
                        unj = 1
                    elif set(r).issubset(set(v2j)):
                        unj = 2
                    elif set(ri).issubset(set(v2j[1:])):
                        unj = 2
                    elif set(ri).issubset(set(v1j[1:])):
                        unj = 1
                        
                    v1igs = hfunc(copi, ui1, ui2, pari, un = uni) 
                    v2jgs = hfunc(copj, uj1, uj2, parj, un = unj)
                    ktau.append(abs(st.kendalltau(v1igs ,  v2jgs  )[0]))
                    #rhos.append(GaussianCopulafit(np.vstack((v1igs ,  v2jgs )).T))
                    v1_k.append(v1igs)
                    v2_k.append(v2jgs)
                       # node.append(li + lj + ['g'] + ri + rj)
                       
                   
                    del uj1,ui1,ui2,uj2
                    
                    node.append(l + ['g'] + r)
                    lk.append(l)
                    rk.append(r)
            orderk =  pd.DataFrame({'v1': v1, 'v2': v2, 'tauabs': ktau, 'node': node, 'l': lk, 'r': rk}  )
           
            for w in range(len(orderk)):
                for q in orderk.l[w]: 
                    if q in orderk.r[w]:
                        orderk = orderk.drop(labels=w, axis=0)
                        break
            if len(orderk) > dimen-k:
                orderk = orderk.sort_values(by='tauabs', ascending=False)
                orderk = orderk.reset_index(drop=True)
                indexes = list(orderk.index)
                inde = []
                inde2 = []
                for i in range(len(orderk)):
                    if i == 0:
                        order = orderk.head(1)
                        inde.append(indexes[i])
                        l = orderk.l[i]
                        
                    else:
                        if ((vint in l) and (vint in orderk.l[i])) or ((orderk.v1[i] in list(orderk.v2[:i]) or orderk.v1[i] in list(orderk.v1[:i]) ) and ( orderk.v2[i] in list(orderk.v2[:i]) or orderk.v2[i] in list(orderk.v1[:i]))):
                            continue
                        else:
                            inde.append(indexes[i])
                            inde2.append(i)
                            order = pd.concat([order, orderk.loc[i].to_frame().T], ignore_index=True)
                            l = l + orderk.l[i]
                if len(order) < (dimen-k):
                    for i in range(1, len(orderk)):
                        if i in inde:
                            continue
                        if ((vint in l) and (vint in orderk.l[i])):
                            continue
                        lst = list(order.v2.astype(str)[order.v1.astype(str) == str(orderk.v1[i])]) + list(order.v1.astype(str)[order.v2.astype(str) == str(orderk.v1[i])])+ list(order.v2.astype(str)[order.v1.astype(str) == str(orderk.v2[i])]) + list(order.v1.astype(str)[order.v2.astype(str) == str(orderk.v2[i])])
                        if len(lst) == len(set(lst)):
                            order = pd.concat([order, orderk.loc[i].to_frame().T], ignore_index=True)
                            inde.append(indexes[i])
                        if len(order) == dimen-k:
                            break
                orderk = order
                orderk = orderk.sort_values(by='tauabs', ascending=False)
                
                v1_k = np.array([v1_k[ind] for ind in inde]).T
                v2_k = np.array([v2_k[ind] for ind in inde]).T
                orderk = orderk.reset_index(drop=True)
                orderk['tree'] = k - 1
        
                for j in range(len(orderk)):
                    u3=np.vstack((v1_k[:,j] , v2_k[:,j])).T
                    cop, rho, aic = bestcop(copsi, u3)
                    aics.append(aic)
                    rhos.append(rho)
                    cops.append(cop)
                
                orderk['rhos'] = rhos
                orderk['cop'] = cops
                orderk['AIC'] = aics
                locals()['v1_' + str(k)] = v1_k
                locals()['v2_' + str(k)] = v2_k
                locals()['order' + str(k)] = orderk
                
            else:
                orderk = orderk.sort_values(by='tauabs', ascending=False)
               
                v1_k = np.array([v1_k[ind] for ind in orderk.index]).T
                v2_k = np.array([v2_k[ind] for ind in orderk.index]).T
                orderk = orderk.reset_index(drop=True)
                orderk['tree'] = k - 1
                    
                for j in range(len(orderk)):
                    u3=np.vstack((v1_k[:,j] , v2_k[:,j])).T
                    cop, rho, aic = bestcop(copsi, u3)
                    rhos.append(rho)
                    aics.append(aic)
                    cops.append(cop)
                    
                orderk['rhos'] = rhos
                orderk['cop'] = cops
                orderk['AIC'] = aics
                locals()['v1_' + str(k)] = v1_k
                locals()['v2_' + str(k)] = v2_k
                locals()['order' + str(k)] = orderk
        
        
        
            
        

        
        
        

        order = pd.DataFrame(columns  = order1.columns)
        for i in range(1,dimen):
            order = pd.concat([order, locals()['order' + str(i)]]).reset_index(drop=True)

            

        a = np.empty((dimen,dimen))
        c = np.empty((dimen,dimen))
        a[:] = np.nan
        c[:] = np.nan
       


        order['used'] = 0
        for i in list(range(dimen-1))[::-1]:
            k1 = sorted(np.array(order[(order.tree == i) & (order['used'] == 0)].node.iloc[0][:2]).astype(int))[::-1]
            order.loc[(order['tree'] == i) & (order['used'] == 0), 'used'] = 1
            t1 = i - 1
            ii = dimen-2-i
            a[i:dimen-ii,ii] = k1
            s = k1[-1]
            for j in list(range(0,i))[::-1]:
                orde = order[(order.tree == j) & (order['used'] == 0)]
                for k in range(len(orde)):
                    arr = np.array(orde.node.iloc[k][:2]).astype(int)
                    if np.isin(s,arr) == True:
                        inde = orde.iloc[k].name
                        a[j, ii] = arr[arr!=s][0]
                        order['used'][inde] = 1

        a[0,dimen-1] = a[0,dimen-2] 
        orderk = pd.DataFrame(columns  = order.columns)
        p = np.empty((dimen,dimen))
        p[:] = np.nan
        p  = p.astype(object)
        for i in list(range(dimen-1)):
            orde = order[order.tree == i]
            for k in list(range(dimen-1-i)):
                ak = a[:,k]
                akn = np.array([ak[-1-k], ak[i]]).astype(int)
                for j in range(len(orde)):
                    arr = np.array(orde.node.iloc[j][:2]).astype(int)
                    if sum(np.isin(akn,arr)) == 2:
                        orderj = order.loc[[orde.index[j]]]
                        p[i,k] = orderj.rhos.iloc[0]
                        c[i,k] =  orderj.cop.iloc[0]
                        if i == 0:
                            orderj.node.iloc[0] = list(akn )
                        else:
                            orderj.node.iloc[0] = list(akn)  + ['|'] + list((ak.astype(int)[:i])[::-1])
                        orderk = pd.concat([orderk, orderj]).reset_index(drop=True)
                        
                        

        for i in list(range(dimen-1)):
            orde = orderk[orderk.tree == i].reset_index(drop=True)
            print('** Tree: ', i)
            for j in range(len(orde)):
                print(orde.node[j], copulas[int(orde.cop[j])], ' ---> parameters = ', orde.rhos[j])
        return a, p, c 


#%% Sampling vine copula
def samplecop(a, p,  c, s):
    """
    Generate random samples from an R-vine.
    
    Arguments:
        
        *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer reffers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.
        
        *p* : Parameters of the bivariate copulae provided as a triangular matrix.
        
        *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers reffering to the copulae with the best fit. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
        
        *s* : number of samples to generate, provided as a positive scalar integer.
        
       
     
     Returns:  
         *X2* :  the randomly sampled data data, provided as a numpy array where each column contains samples of a seperate variable (eg. u1,u2,...,un).
         
    """
    # Reference: Dißmann et al. 2012 
    Ms =  np.flipud(a)
    P =   np.flipud(p)
    C =   np.flipud(c)
    replace = {}
    for i in range(int(max(np.unique(Ms))+1)):
        val = max(np.unique(Ms))- i
       # Ms[k,k]
        replace.update({Ms[i,i]: val})
        
    Ms = np.nan_to_num(Ms, nan=int(max(np.unique(Ms))+1))

    # Create a vectorized function for replacement
    replace_func = np.vectorize(lambda x: replace.get(x, x))

    # Apply the replacement to the array
    M = replace_func(Ms)
    M[M==np.max(Ms)]=np.nan
    Mm = M.copy()
    #i = row,k = column
    for i in range(M.shape[0]):
        for k in range(M.shape[0]):
            if k == i:
                continue
            if i == 0:
                continue
            Mm[i,k] = max(Mm[i:,k])
            
            
            

    Vdir = np.empty((s, M.shape[0],M.shape[0]))
    Vdir[:] = np.nan
    Vindir =  np.empty((s, M.shape[0],M.shape[0]))
    Vindir[:] = np.nan
    Z2 = np.empty((s, M.shape[0],M.shape[0]))
    Z2[:] = np.nan
    Z1 = np.empty((s, M.shape[0],M.shape[0]))
    Z2[:] = np.nan
    U = np.random.uniform(0,1,(s, M.shape[0]))
    Vdir[:,-1,:] = U.copy()
    X =np.flip(U.copy(), 1)
    n = M.shape[0]-1
    for k in range(n)[::-1]:
        for i in range(k+1, n+1):
            if M[i,k] == Mm[i,k]:
                Z2[:,i,k] = Vdir[:,i, int(n- Mm[i,k])]
            else:
                Z2[:,i,k] = Vindir[:,i, int(n- Mm[i,k])]
            #Vdir[:,n,k] =  inverse_condcdfgaus(Z2[:,i,k], Vdir[:,n,k], P[i,k])
            Vdir[:,n,k] =   hfuncinverse( int(C[i,k]), Z2[:,i,k],Vdir[:,n,k],  P[i,k], un = 2)
        X[:,int(n-k)] = Vdir[:,n, k]
        for i in range(k+1, n+1)[::-1]:
            Z1[:,i,k] = Vdir[:,i,k]
            #Vdir[:,int(i-1),k] =  condcdfgaus(Z1[:,i,k],Z2[:,i,k],P[i,k])
            #Vindir[:,int(i-1),k] =  condcdfgaus(Z2[:,i,k],Z1[:,i,k],P[i,k])
            Vdir[:,int(i-1),k] =  hfunc(int(C[i,k]), Z1[:,i,k],Z2[:,i,k], P[i,k], un = 2)
            Vindir[:,int(i-1),k] =  hfunc(int(C[i,k]), Z1[:,i,k],Z2[:,i,k], P[i,k], un = 1)
            
            
            
    replacedf = pd.DataFrame(list(replace.items()), columns=['Original', 'Replacement'])
    replacedf = replacedf.sort_values(by='Original')
    X2 = np.array([])
    for i in replacedf.Replacement:
        if len(X2) == 0:
            X2 =X[:,int(i)].reshape(len(X),1)
        else:
            X2 = np.hstack((X2, X[:,int(i)].reshape(len(X),1)))   
    return X2

#%% Sampling conditonal vine copula
def vincopconditionalsample(a, p,c, s, Xc):
   """
   Generate conditional samples from an R-vine based on a provided sampling order
   
   Arguments:
       
       *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer reffers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.
       
       *p* : Parameters of the bivariate copulae provided as a triangular matrix.
       
       *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers reffering to the copulae with the best fit. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
       
       *s* : number of samples to generate, provided as a positive scalar integer.
       
       *XC*: the values of the variables on which the conditional sample has to generated, provided as a 1d array that contains the the values ordered in terms of the sampling order. 
    
    Returns:  
        *X2* :  the randomly sampled data data, provided as a numpy array where each column contains samples of a seperate variable (eg. u1,u2,...,un).
        
   """ 
   Ms =  np.flipud(a)
   P =   np.flipud(p)
   C =   np.flipud(c)
   replace = {}
   for i in range(int(max(np.unique(Ms))+1)):
       val = max(np.unique(Ms))- i
      # Ms[k,k]
       replace.update({Ms[i,i]: val})
       
   Ms = np.nan_to_num(Ms, nan=int(max(np.unique(Ms))+1))

   # Create a vectorized function for replacement
   replace_func = np.vectorize(lambda x: replace.get(x, x))

   # Apply the replacement to the array
   M = replace_func(Ms)
   M[M==np.max(Ms)]=np.nan
   Mm = M.copy()
   #i = row,k = column
   for i in range(M.shape[0]):
       for k in range(M.shape[0]):
           if k == i:
               continue
           if i == 0:
               continue
           Mm[i,k] = max(Mm[i:,k])
           
           

   Vdir = np.empty((s, M.shape[0],M.shape[0]))
   Vdir[:] = np.nan
   Vindir =  np.empty((s, M.shape[0],M.shape[0]))
   Vindir[:] = np.nan
   Z2 = np.empty((s, M.shape[0],M.shape[0]))
   Z2[:] = np.nan
   Z1 = np.empty((s, M.shape[0],M.shape[0]))
   Z2[:] = np.nan

   U =np.hstack((np.random.uniform(0,1, (s, M.shape[0]-len(Xc))), np.flip(np.tile(Xc, (s, 1)).copy(), 1)))
   Vdir[:,-1,:] = U.copy()
   X =np.flip(U.copy(), 1)
   n = M.shape[0]-1






   for k in range(n)[::-1]:    
       for i in range(k+1, n+1):
           if M[i,k] == Mm[i,k]:
               Z2[:,i,k] = Vdir[:,i, int(n- Mm[i,k])]
           else:
               Z2[:,i,k] = Vindir[:,i, int(n- Mm[i,k])]
           if k <= n-len(Xc):
            #   Vdir[:,n,k] =  inverse_condcdfgaus(Z2[:,i,k], Vdir[:,n,k], P[i,k])
               Vdir[:,n,k] =   hfuncinverse( int(C[i,k]), Z2[:,i,k],Vdir[:,n,k],  P[i,k], un = 2)
       X[:,int(n-k)] = Vdir[:,n, k]
       for i in range(k + 1, n + 1)[::-1]:
          Z1[:,i,k] = Vdir[:,i,k]
         # Vdir[:,int(i-1),k] =  condcdfgaus(Z1[:,i,k],Z2[:,i,k],P[i,k])
         # Vindir[:,int(i-1),k] =  condcdfgaus(Z2[:,i,k],Z1[:,i,k],P[i,k])
          Vdir[:,int(i-1),k] =  hfunc(int(C[i,k]), Z1[:,i,k],Z2[:,i,k], P[i,k], un = 2)
          Vindir[:,int(i-1),k] =  hfunc(int(C[i,k]), Z1[:,i,k],Z2[:,i,k], P[i,k], un = 1)
          
   replacedf = pd.DataFrame(list(replace.items()), columns=['Original', 'Replacement'])
   replacedf = replacedf.sort_values(by='Original')
   X2 = np.array([])
   for i in replacedf.Replacement:
       if len(X2) == 0:
           X2 =X[:,int(i)].reshape(len(X),1)
       else:
           X2 = np.hstack((X2, X[:,int(i)].reshape(len(X),1)))   
   return X2
#%%sampling orders

def samplingorder(a, c, p):
    
    """
    Provides all the different sampling orders that are possible for the fitted vine-copula.
    
    Arguments:
        
        *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer reffers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.
        
        *p* : Parameters of the bivariate copulae provided as a triangular matrix.
        
        *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers reffering to the copulae with the best fit. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
        
       
     
     Returns:  
         *sortingorder* :  A list of the different sampling orders available for the fitted vine-copula
         
    """
    
    dimen = a.shape[0]
    order = pd.DataFrame(columns  = ['node', 'par', 'cop', 'l', 'r', 'tree'])
    s = 0
    for i in list(range(dimen-1)):
        for k in list(range(dimen-1-i)):
            ak = a[:,k]
            akn = np.array([ak[-1-k], ak[i]]).astype(int)
            if i == 0:
                single_row_values = {
                    'node': list(akn),
                    'par': p[i,k],
                    'cop': int(c[i,k]),
                    'l': akn[0],
                    'r': akn[1],
                    'tree': i
                }
            else:
                single_row_values = {
                    'node': list(akn)  + ['|'] + list((ak.astype(int)[:i])[::-1]),
                    'par': p[i,k],
                    'cop': int(c[i,k]),
                    'l': list(akn),
                    'r': list((ak.astype(int)[:i])[::-1]),
                    'tree': i
                }
        
            order.loc[s] = single_row_values
            s = s +1
    combinations = list(product([True, False], repeat=dimen-1))
    sortingorder = []
    for q in combinations:
        a = np.empty((dimen,dimen))
        a[:] = np.nan
        order['used'] = 0
        for i in list(range(dimen-1))[::-1]:
            k1 = sorted(np.array(order[(order.tree == i) & (order['used'] == 0)].node.iloc[0][:2]).astype(int), reverse=q[i])
            order.loc[(order['tree'] == i) & (order['used'] == 0), 'used'] = 1
            ii = dimen-2-i
            a[i:dimen-ii,ii] = k1
            s = k1[-1]
            for j in list(range(0,i))[::-1]:
                orde = order[(order.tree == j) & (order['used'] == 0)]
                for k in range(len(orde)):
                    arr = np.array(orde.node.iloc[k][:2]).astype(int)
                    if np.isin(s,arr) == True:
                        inde = orde.iloc[k].name
                        a[j, ii] = arr[arr!=s][0]
                        order['used'][inde] = 1
        a[0,dimen-1] = a[0,dimen-2] 
        sortingorder.append(list(np.diag(a[::-1])[::-1])) 
    return sortingorder
            
#%%matrices of specific sampling order

def samplingmatrix(a,c,p,sorder):
    """
    Provides the triangular matrices for which the samples can be generated based on the specific sampling order.
    
    Arguments:
        
        *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer reffers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.
        
        *p* : Parameters of the bivariate copulae provided as a triangular matrix.
        
        *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers reffering to the copulae with the best fit. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
        
        *sorder* :  A list containing the specific sampling order of interest
     
     Returns:  
         *ai* : The vine tree structure, based on the sampling order, provided as a triangular matrix, composed of integers. The integer reffers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.
         
         *pi* : Parameters of the bivariate copulae, based on the sampling order,  provided as a triangular matrix.
         
         *ci* : The types of the bivariate copulae, based on the sampling order,  provided as a triangular matrix, composed of integers reffering to the copulae with the best fit. eg. a 1 reffers to the gaussian copula (see...reffer to where this information would be)
         
    """
    
    dimen = a.shape[0]
    order = pd.DataFrame(columns  = ['node', 'par', 'cop', 'l', 'r', 'tree'])
    s = 0
    for i in list(range(dimen-1)):
        for k in list(range(dimen-1-i)):
            ak = a[:,k]
            akn = np.array([ak[-1-k], ak[i]]).astype(int)
            if i == 0:
                single_row_values = {
                    'node': list(akn),
                    'par': p[i,k],
                    'cop': int(c[i,k]),
                    'l': akn[0],
                    'r': akn[1],
                    'tree': i
                }
            else:
                single_row_values = {
                    'node': list(akn)  + ['|'] + list((ak.astype(int)[:i])[::-1]),
                    'par': p[i,k],
                    'cop': int(c[i,k]),
                    'l': list(akn),
                    'r': list((ak.astype(int)[:i])[::-1]),
                    'tree': i
                }
        
            order.loc[s] = single_row_values
            s = s +1
    ai = np.empty((dimen,dimen))
    ai[:] = np.nan
    order['used'] = 0
    ci = np.empty((dimen,dimen))
    ci[:] = np.nan
    pi = np.empty((dimen,dimen))
    pi[:] = np.nan
    pi = pi.astype(object)
    for i in list(range(dimen-1))[::-1]:
        if i == 0:
            k1 = order[(order.tree == i) & (order['used'] == 0)].node.iloc[0]
        else:
            k1 = order[(order.tree == i) & (order['used'] == 0)].l.iloc[0]
        
        if sorder[i + 1] == max(k1):
            k1 = sorted(k1, reverse=False)
        else:
            k1 = sorted(k1, reverse=True)
        ii = dimen-2-i
        ai[i:dimen-ii,ii] = k1
        ci[i, ii] =  order[(order.tree == i) & (order['used'] == 0)].cop.iloc[0]
        pi[i, ii] = order[(order.tree == i) & (order['used'] == 0)].par.iloc[0]
        order.loc[(order['tree'] == i) & (order['used'] == 0), 'used'] = 1
        s = k1[-1]
        for j in list(range(0,i))[::-1]:
            orde = order[(order.tree == j) & (order['used'] == 0)]
            for k in range(len(orde)):
                arr = np.array(orde.node.iloc[k][:2]).astype(int)
                if np.isin(s,arr) == True:
                    inde = orde.iloc[k].name
                    ai[j, ii] = arr[arr!=s][0]
                    ci[j, ii] = order['cop'][inde]
                    pi[j, ii] = order['par'][inde]
                    order['used'][inde] = 1
    ai[0,dimen-1] = ai[0,dimen-2]
    return ai, pi, ci

#%% best fit discrete distribution

def best_fit_distributiondiscrete(data, bound = False):
    """Model data by finding best fit distribution to data"""
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

    
    return sorted(best_distributions, key=lambda x:x[2])

#%% best fit distribution

def best_fit_distribution(data):
    """Model data by finding best fit distribution to data"""
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