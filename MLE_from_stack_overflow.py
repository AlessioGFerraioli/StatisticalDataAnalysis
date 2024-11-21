# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:49:09 2020

MLE in python, preso da:
    https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-9fbff27ea12f
    
@author: aless
"""

# import libraries
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats
#import pymc3 as pm3
#import numdifftools as ndt
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
#%matplotlib inline

'''
From there, we will generate data that follows a normally distributed 
errors around a ground truth function:
'''

# generate data
N = 100
x = np.linspace(0,20,N)
ϵ = np.random.normal(loc = 0.0, scale = 5.0, size = N)   #gaussian noise
y = 3*x + ϵ
df = pd.DataFrame({'y':y, 'x':x})
df['constant'] = 1

'''
 let’s visualize using Seaborn’s regplot:
'''

# plot
#sns.regplot(df.x, df.y);

#%%

# split features and target
X = df[['constant', 'x']]
# fit model and summarize (with ordinary least square method)
sm.OLS(y,X).fit().summary()

#%%

# now we do the same fit but with the maximum likelihood method that should
# give the same result

# define likelihood function
def MLERegression(params):
    intercept, beta, sd = params[0], params[1], params[2] # inputs are guesses at our parameters
    yhat = intercept + beta*x # predictions
    # next, we flip the Bayesian question
    # compute PDF of observed values normally distributed around mean (yhat)
    # with a standard deviation of sd
    negLL = -np.sum( stats.norm.logpdf(y, loc=yhat, scale=sd) )
    # return negative LL
    return(negLL)
    
    
'''
Now that we have a cost function, let’s initialize and minimize it:
'''    

# let’s start with some random coefficient guesses and optimize
guess = np.array([5,5,2])
results = minimize(MLERegression, guess, method = 'Nelder-Mead', 
                   options={'disp': True})

# drop results into df and round to match statsmodels
resultsdf = pd.DataFrame({'coef':results['x']})
resultsdf.index=['constant','x','sigma']   
np.round(resultsdf.head(2), 4)