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
def plot_overlays(x, y, image, labels, title=None): #x,y are touples
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    fig.suptitle(title)
    
    if isinstance(x,tuple):
        for i in range(len(x)):
            # plot original xy
            ax[0].plot(x[i], y[i], label=labels[i])
            ax[1].plot(x[i], y[i], label=labels[i])

    ax[1].imshow(image, origin='upper', cmap='gray')
    ax[0].legend()
#------- Should be in Helper Scripts graphical -------

def normal_equ_w(design_matrix, y):
    return np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ y  

def normal_equ_var(X, w, y):
    N = len(X)
    return 1/N * (y- (X @ w)).T @ (y- (X @ w))

def output(design_matrix, y_or_w, predict=False):
    if predict==False:  
        y = y_or_w
        w = normal_equ_w(design_matrix, y)
        var = normal_equ_var(design_matrix, w, y)# what do with this?
        out = w.T @ design_matrix.T
        return out, var, w
    else:
        w = y_or_w
        return w.T @ design_matrix.T
    

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
    

def score(pred, y):
    return np.sum((pred-y)**2)**0.5


def polynomial_splining(splines, max_poly=9, num_new_x=1000, plot_examples=False):

    for key in splines.keys(): # for each file
        im = io.imread(ts_path+'data/Processed/'+key+'_ZSensor_processed_grey.tif')
        for mol in splines[key]['Splines']: # for each molecule in file
            score_list = []
            x = np.asarray(splines[key]['Splines'][mol]['x_coord'])
            y = np.asarray(splines[key]['Splines'][mol]['y_coord'])
            for i in range(1, max_poly+1): # for each polynomial degree
                # Compute a design matrix, run through ML and compute a score
                basis = basis_func.polynomial(x,i)
                pred = output(basis, y)[0]
                score_list.append(score(pred,y))
            # Get weights from best score polynomial
            basis_num = score_list.index(min(score_list))+1 # +1 as start at poly 1 which mismatches with list index
            basis = basis_func.polynomial(x, basis_num)
            preds, var, w = output(basis, y, predict=False)
            # Get predictions on weights for more x values
            new_x = np.linspace(min(x), max(x), 1000)
            new_basis = basis_func.polynomial(new_x, basis_num)
            splined_preds = output(new_basis, w, predict=True)
            
            # Plot a graph
            if plot_examples==True:
                title='Polynomial Degree = %i'%basis_num
                x_tup, y_tup = (x,x,new_x), (y, preds, splined_preds)
                labels = ('Topo', 'Poly', 'Poly Splined')
                plot_overlays(x_tup, y_tup, im, title=title, labels=labels)#, new_x=new_x)
                plt.imshow(im, cmap='gray')
                plt.legend()
            # Overwrite current x and y coords
            splines[key]['Splines'][mol]['x_coord'] = new_x
            splines[key]['Splines'][mol]['y_coord'] = splined_preds

#------- Run Script --------
ts_path = "/Users/Maxgamill/Desktop/Uni/PhD/TopoStats/"
splines = get_spline_jsons(ts_path + 'data/')
im = io.imread(ts_path+'data/Processed/20161024_339_LIN_6ng_434rep_8ng_PLL.026_ZSensor_processed_grey.tif')

polynomial_splining(splines, max_poly=9, num_new_x=1000, plot_examples=True)
