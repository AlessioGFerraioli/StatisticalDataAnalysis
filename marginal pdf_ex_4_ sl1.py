# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:35:35 2020

@author: aless
"""

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x+y

X = np.linspace(0,1,100)
Y = np.linspace(0,1,100)


Z = f(X, Y) 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)

