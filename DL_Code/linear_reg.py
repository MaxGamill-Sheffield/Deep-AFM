#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:29:32 2021

@author: Maxgamill
"""

import numpy as np


def normal_equ_w(design_matrix, y):
    return np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ y  

def normal_equ_var(X, w, y):
    N = len(X)
    return 1/N * (y- (X @ w)).T @ (y- (X @ w))

def output(design_matrix, y):
    w = normal_equ_w(design_matrix, y)
    var = normal_equ_var(design_matrix, w, y)
    out = w.T @ design_matrix.T #+ var
    return out

class basis_func():
    
    def polynomial(x, n):
        return x**n
    
    def exp(x, s=1):
        return np.exp(-1*(x-np.mean(x))**2/(2*s)**2)
    
    def sigmoid(x, s=1):
        1/(1+np.exp(-1*(x-np.mean(x))/s))

    def gaussian_basis(x, var):
        return np.exp(-1 * ((x-np.mean(x))**2)/var)
 

   
from scikit import io
import matplotlib.pyplot as plt
import Deep_AFM.Helper_Scripts.json_edittor as je

ts_path = "/Users/Maxgamill/Desktop/Uni/PhD/TopoStats/"
splines = je.get_spline_jsons(ts_path + 'data/')
im = io.imread(ts_path+'data/Processed/20160722_339_REL_15ng_434rep_80ng_PLL.077_ZSensor_processed_grey.tif')

lin = splines['20160722_339_REL_15ng_434rep_80ng_PLL.077']['Splines']['2']
x = np.asarray(lin['x_coord'])
#X = np.column_stack([np.ones(len(x)), x]) #including bias
y = np.asarray(lin['y_coord'])

'''
x = np.append([1], np.asarray(lin['x_coord']), axis=0) # with bias
y = np.append([1], np.asarray(lin['y_coord']), axis=0) # with bias
'''

M=4
fig, ax = plt.subplots(1,2)

phi = np.ones((y.shape[0], M))
for m in range(M-1):
    phi[:, m+1] = basis_func.polynomial(x, M)

phi = np.array([basis_func.polynomial(x, n) for n in range(M+1)]).T
ml_out = output(phi, y)

fig.suptitle('ML Gaussian splining with Polynomial Degree M=%i'%M)
label = 'ML Gaussian'
ax[0].plot(x,y, label='TopoStats')
ax[0].plot(x,ml_out, label=label)
ax[1].plot(x,ml_out, color='b')
ax[1].imshow(im, origin='upper', cmap='gray')
ax[0].legend()




