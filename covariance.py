# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:41:04 2020

@author: Alessio Giuseppe Ferraioli
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats


x = np.random.randn(100)*.5
y = np.random.randn(100)*.5

fig, ax = plt.subplots()
ax.scatter(x, y, marker='.')

print("correlation coeff:",scipy.stats.pearsonr(x, y))
