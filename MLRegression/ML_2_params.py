# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:33:22 2020
Exercise from Cowan's book on statistical data analysis.

Generate a virtual experiment of a 2-parameter pdf (that describes the angular
distribution of some particle physics stuff) and then make some ML esitames and
whatnot. vedi Cowan par 6.8

es prof. Sioli sl 4 pag 36

@author: Alessio Giuseppe Ferraioli
"""


# import libraries
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.interpolate as interpolate
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


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


def angular_cdf(x, paramaters):
    ''' cumulative distribution function for the angular pdf, calcolata 
    integrandola analiticamente (carta e penna bro)'''
    alpha, beta = parameters
    num = beta * x * x * x / 3 + alpha * x * x * .5 + x + 1 + beta/3 - alpha/2 
    den = 2 + 2 * beta / 3

    return num/den

'''
Generate linear data with some gaussian noise
'''
# parameters for experiment
n_obs = 2000  #n osservazioni per esperimento
alpha = .5
beta = .5
parameters = alpha, beta
data = inverse_transform_sampling(angular_cdf, parameters, n_obs)
    
# generate data
N = 100
x = np.linspace(0,20,N)
ϵ = np.random.normal(loc = 0.0, scale = 5.0, size = N)   #gaussian noise
y = 3*x + ϵ
df = pd.DataFrame({'y':y, 'x':x})
df['constant'] = 1

# plot data with seaborn automatic regression line (that we will compare to our method)
fig, ax = plt.subplots()
sns.regplot(x = df.x, y=df.y, ax=ax,label="Seaborn regression")
ax.set_title('Simple linear regression')


# split features and target
X = df[['constant', 'x']]
# fit model and summarize (with ordinary least square method)
sm.OLS(y,X).fit().summary()

# now we do the same fit but with the maximum likelihood method that should give the same result
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

print("resultdf")
print(resultsdf)

# extract intercept and slope from results
q = resultsdf.loc['constant', 'coef']
m = resultsdf.loc['x', 'coef']

print("Parameters from Maximum Likelihood Estimation (MLE) method:  ")
print("intercept: ", q)
print("slope: ", m)

# plot the data with the MLE regression line
#fig, ax = plt.subplots()
ax.scatter(x = df.x, y=df.y)
ax.plot(x, q + m*x, color='red', label='MLE regression', linestyle='dashed')
ax.set_title('Maximum Likelihood Estimation (MLE) regression')
ax.legend()
fig.savefig('MLRegression.png')
