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
    
    def polynomial(x, n):
        return x**n
    
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
def plot_poly_overlays(x, y, M):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    phi = np.ones((y.shape[0], M))
    for m in range(M-1):
        phi[:, m+1] = basis_func.polynomial(x, M)
    
    phi = np.array([basis_func.polynomial(x, n) for n in range(M+1)]).T
    ml_out, var_out = output(phi, y)
    
    fig.suptitle('ML Gaussian splining with Polynomial Degree M=%i'%M)
    label = 'ML Gaussian'
    ax[0].plot(x,y, label='TopoStats')
    ax[0].plot(x,ml_out, label=label)
    ax[1].plot(x,y, label='TopoStats')
    ax[1].plot(x,ml_out, label=label)
    ax[1].imshow(im, origin='upper', cmap='gray')
    ax[0].legend()


def plot_exp_overlays(x, y, M):    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    phi = np.ones((y.shape[0], len(M)))
    for m in M:
        phi[:, m+1] = basis_func.exp(x, m)
    
    phi = np.array([basis_func.exp(x, m) for m in M]).T
    ml_out = output(phi, y)
    
    fig.suptitle('ML Exp splining with sigma=%i, mu=%i'%(m,1))
    label = 'ML Exp'
    ax[0].plot(x,y, label='TopoStats')
    ax[0].plot(x,ml_out, label=label)
    ax[1].plot(x,y, label='TopoStats')
    ax[1].plot(x,ml_out, label=label)
    ax[1].imshow(im, origin='upper', cmap='gray')
    ax[0].legend()
#------- Should be in Helper Scripts graphical -------


ts_path = "/Users/Maxgamill/Desktop/Uni/PhD/TopoStats/"
splines = get_spline_jsons(ts_path + 'data/')
im = io.imread(ts_path+'data/Processed/20160722_339_REL_15ng_434rep_80ng_PLL.077_ZSensor_processed_grey.tif')

lin = splines['20160722_339_REL_15ng_434rep_80ng_PLL.077']['Splines']['2']
x = np.asarray(lin['x_coord'])
#X = np.column_stack([np.ones(len(x)), x]) #including bias
y = np.asarray(lin['y_coord'])