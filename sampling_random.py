# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:17:18 2020
sampling from distributions

slide sioli lezione 3 (PRNG) pag 27
@author: aless
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st



def PoissonGen1(mu):
    k=-1
    s = 1
    q = np.exp(-mu)
    while( s > q ):
        r = np.random.rand()
        s = s * r
        k = k + 1
    return k

def BinomialGen1(n, p):
    m = 0
    for i in range(n):
        r = np.random.rand()
        if (r <= p):
            m = m+1
    return m

iterations = 10

# BINOMIAL DISTRIBUTION


binomial_seq = np.zeros(iterations)
n = iterations
p = .2

for i in range(iterations):
    binomial_seq[i]=BinomialGen1(n, p)
fig, ax = plt.subplots(1, 1)
values = np.arange(iterations)
pmf = st.binom(n, p).pmf(values)
ax.bar(values, pmf, alpha=0.5)

   
plt.hist(binomial_seq, color='g',alpha=0.5)
plt.suptitle('Random Sampling from Binomial Distribution')
plt.title(f'n iterations: {iterations}, p: {p}')
plt.savefig('binomial_dist.png')
print("Salvato plot binomial_dist.png\n")

# POISSON DISTRIBUTION 


poisson_seq = np.zeros(iterations)
mu = .9
for i in range(iterations):
    poisson_seq[i]=PoissonGen1(mu)


fig, ax = plt.subplots(1, 1)
values = np.arange(20)
pmf = st.poisson(mu).pmf(values)
ax.bar(values, pmf, alpha=0.5)
 
plt.hist(poisson_seq, color='g', alpha=0.5)
plt.suptitle('Random Sampling from Poisson Distribution')
plt.title(f'n iterations: {iterations}, mu: {mu}')
plt.savefig('poisson_dist.png')
print("Salvato plot poisson_dist.png\n")