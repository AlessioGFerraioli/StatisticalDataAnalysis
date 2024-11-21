# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:24:11 2020

generate a virtual experiment of a 2-parameter pdf (that describes the angular
distribution of some particle physics stuff) and then make some ML esitames and
whatnot. vedi Cowan par 6.8

es sioli sl 4 pag 36
@author: aless
"""

import numpy as np, pandas as pd
import matplotlib.pylab as plt
import scipy.interpolate as interpolate
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats
#import pymc3 as pm3
#import numdifftools as ndt
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

#%%
np.random.seed(122546)


def inverse_transform_sampling(cdf, parameters, n_samples=1000):
    ''' 
    this function does the inverse transform sampling of a pdf given its cdf
    
    cdf : cumulative distribution function of the pdf we want to sample from
    parameters : parameters of the cdf function
    n_samples : number of samples we want to generate
    
    it works by creating a vector of x uniformly distributed values and 
    evaluating the cdf(x). in this way we build the cdf=cdf(x) function by 
    points. it then finds the inverse of this function, that is x=x(cdf) by
    using a scipy interpolation (which returns a function that interpolates 
    the point based on the point it is been given - in our case cdf(x) and x)
    
    strongly inspired by (i.e. copied from):
        https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
        '''
    
    n_x = 100     
    x  = np.linspace(-1, 1, n_x)
    cum_values = np.zeros(n_x)
    for i in range(n_x):
        cum_values[i] = cdf(x[i], parameters)
    inv_cdf = interpolate.interp1d(cum_values, x)
    r = np.random.rand(n_samples)
    return inv_cdf(r)

def angular_pdf(parameters):
    ''' pdf, vedi cowan 6.8'''
    alpha, beta = parameters
    x = (np.random.rand() * 2) - 1   # r.v. between -1 and 1
    num = 1 + alpha * x + beta * x * x
    den = 2 + 2 * beta / 3
    return num/den

def angular_cdf(x, paramaters):
    ''' cumulative distribution function for the angular pdf, calcolata 
    integrandola analiticamente (carta e penna bro)'''
    alpha, beta = parameters
    num = beta * x * x * x / 3 + alpha * x * x * .5 + x + 1 + beta/3 - alpha/2 
    den = 2 + 2 * beta / 3

    return num/den



n_obs = 2000  #n osservazioni per esperimento
n_experiments = 5 
experiment = np.zeros((n_experiments, n_obs))

alpha = .5
beta = .5
parameters = alpha, beta
for i in range(n_experiments):
    experiment[i] = inverse_transform_sampling(angular_cdf, parameters, n_obs)
    



#%%
for i in range(max(n_experiments, 10)):
    plt.hist(experiment[i], bins=50, alpha=.2)  