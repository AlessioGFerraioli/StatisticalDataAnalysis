# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:17:18 2020
sampling from distributions

prof. sioli lezione 3 (PRNG) pag 27
@author: Alessio Giuseppe Ferraioli
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st



def PoissonGen1(mu):
    # draws a random value from a Poisson distribution
    k=-1
    s = 1
    q = np.exp(-mu)
    while( s > q ):
        r = np.random.rand()
        s = s * r
        k = k + 1
    return k

def BinomialGen1(n, p):
    # draws a random value from a Poisson distribution
    m = 0
    for i in range(n):
        r = np.random.rand()
        if (r <= p):
            m = m+1
    return m

iterations = 1000


# EXAMPLE OF SAMPLING FROM BINOMIAL DISTRIBUTION
binomial_seq = np.zeros(iterations)
n = iterations
p = .2 # parameter for the Binomial distribution

# generate a sequence of random values from the Binomial distribution
for i in range(iterations):
    binomial_seq[i]=BinomialGen1(n, p)

# plot the generated values and compare it to the "real" Binomial distribution
fig, ax = plt.subplots(1, 1)
values = np.arange(iterations)
pmf = st.binom(n, p).pmf(values)
ax.bar(values, pmf*iterations*10, alpha=0.2)

   
plt.hist(binomial_seq, color='g',alpha=0.3)
plt.suptitle('Random Sampling from Binomial Distribution')
plt.title(f'n iterations: {iterations}, p: {p}')
plt.savefig('RandomSampling/binomial_dist.png')
print("Salvato plot binomial_dist.png\n")




# EXAMPLE OF SAMPLING FROM POISSON DISTRIBUTION


poisson_seq = np.zeros(iterations)
mu = .9 # parameter for the Poisson distribution
# generate a sequence of random values from the Poisson distribution
for i in range(iterations):
    poisson_seq[i]=PoissonGen1(mu)

fig, ax = plt.subplots(1, 1)
values = np.arange(20)
pmf = st.poisson(mu).pmf(values)
ax.bar(values, pmf*iterations, alpha=0.3)
 
ax.hist(poisson_seq, color='g', alpha=0.3)
plt.suptitle('Random Sampling from Poisson Distribution')
plt.title(f'n iterations: {iterations}, mu: {mu}')
fig.savefig('RandomSampling/poisson_dist.png')
print("Salvato plot poisson_dist.png\n")