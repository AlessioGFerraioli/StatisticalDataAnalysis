# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:35:35 2020
Exercise: 

consider the joint probability density 
for two continuous variables x and y
given by:

f(x,y) = x + y     if 0 <= x <= 1 and 0 <= y <= 1
          0         otherwise

Find the marginal pdf probability

@author: Alessio Giuseppe Ferraioli
"""
#%%
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    # define the joint probability density function
    return x+y

X = np.linspace(0,1,100)
Y = np.linspace(0,1,100)

Z = f(X, Y) 
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)
fig.savefig('marginal_pdf.png')

# %%
