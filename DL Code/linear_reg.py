#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:29:32 2021

@author: Maxgamill
"""

import numpy as np
import json
import os
from skimage import io
import matplotlib.pyplot as plt


def normal_equ_w(design_matrix, y):
    return np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ y  

def normal_equ_var(X, w, y):
    N = len(X)
    return 1/N * (y- (X @ w)).T @ (y- (X @ w))

def output(design_matrix, y):
    w = normal_equ_w(design_matrix, y)
    var = normal_equ_var(design_matrix, w, y)# what do with this?
    out = w.T @ design_matrix.T
    return out, var

class basis_func():
    
    def polynomial(x, num_basis=4, data_limits=[-1., 1.]):
        "Polynomial basis"
        centre = data_limits[0]/2. + data_limits[1]/2.
        span = data_limits[1] - data_limits[0]
        z = (x / x.mean()) * span + centre
        z = z.reshape(len(z),1)
        Phi = np.zeros((x.shape[0], num_basis))
        for i in range(num_basis):
            Phi[:, i:i+1] = z**i
        return Phi
    
    def sigmoid(x, s=1):
        1/(1+np.exp(-1*(x-np.mean(x))/s))

    def gaussian(x, var):
        return np.exp(-1 * ((x-np.mean(x))**2)/var)
 
#------- In Helper Scripts -------
def read_json(path):
    with open(path) as file:
        return json.load(file)

def get_spline_jsons(path):
    spline_dict = {}
    filelist = os.listdir(path)
    for file in filelist[:]:
        if not(file.endswith('.json')):
            filelist.remove(file)
        else:
            name = file.split('_spline')[0]
            data = read_json(path + file)
            spline_dict[name] = data
    return spline_dict
#------- In Helper Scripts -------

#------- Should be in Helper Scripts graphical -------
def plot_overlays(x, x_pred, y, title=None):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    fig.suptitle(title)
    ax[0].plot(x, y, label='TopoStats Fit')
    ax[0].plot(x, x_pred, label='ML Fit')
    ax[1].plot(x, y, label='TopoStats Fit')
    ax[1].plot(x, x_pred, label='ML Fit')
    ax[1].imshow(im, origin='upper', cmap='gray')
    ax[0].legend()

#------- Should be in Helper Scripts graphical -------


ts_path = "/Users/Maxgamill/Desktop/Uni/PhD/TopoStats/"
splines = get_spline_jsons(ts_path + 'data/')
im = io.imread(ts_path+'data/Processed/20161024_339_LIN_6ng_434rep_8ng_PLL.026_ZSensor_processed_grey.tif')

lin = splines['20161024_339_LIN_6ng_434rep_8ng_PLL.026']['Splines']['1']

x = np.asarray(lin['x_coord'])
y = np.asarray(lin['y_coord'])

M=6

basis = basis_func.polynomial(x,M)
pred, var = output(basis, y)
title='Polynomial Degree = %i'%M
plot_overlays(x, pred, y, title)

plt.imshow(im, cmap='gray')
plt.legend()
plt.show()