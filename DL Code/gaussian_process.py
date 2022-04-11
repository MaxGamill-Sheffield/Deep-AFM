#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:38:50 2022

@author: Maxgamill
"""

import numpy as np
import json
import os
from skimage import io
import matplotlib.pyplot as plt
import math

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
def plot_overlays(x, y, image, labels, title=None, std=None): #x,y are touples
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    fig.suptitle(title)
    colors = ['m','c','g']
    x_offset, y_offset = min(x[0]-20), min(y[0]-20)
    
    ax[1].imshow(image[y_offset:max(y[0]+20), x_offset:max(x[0]+20)])
    
    if isinstance(x,tuple):
        for i in range(len(x)):
            # plot original xy
            ax[0].plot(x[i]-x_offset, y[i]-y_offset, label=labels[i], color=colors[i])
            ax[1].plot(x[i]-x_offset, y[i]-y_offset, label=labels[i], color=colors[i])
            try:
                ax[0].fill_between(x[i]-x_offset, (y[i]-y_offset-std[i]*3), (y[i]-y_offset+std[i]*3), alpha=0.1)
                ax[1].fill_between(x[i]-x_offset, (y[i]-y_offset-std[i]*3), (y[i]-y_offset+std[i]*3), alpha=0.1)
            except:
                pass
                
    ax[0].set_ylim(ax[0].get_ylim()[::-1])
    
    ax[0].legend()
    ax[1].legend()
#------- Should be in Helper Scripts graphical -------

class GP():
    
    def __init__(self, x_train, y_train, kernel, noise_var=1e-6, samples=10, kernel_l=1, kernel_sigma=1):
        self.noise_var = noise_var
        self.samples = samples
        self.N = len(x_train) # number of test points
        self.x = x_train # given x values in vertical array
        self.y = y_train # given x values in vertical array
        self.K = kernel(self.x, self.x, kernel_l, kernel_sigma)
        self.L = np.linalg.cholesky(self.K + self.noise_var * np.eye(self.N))
        self.prior = np.dot(self.L, np.random.normal(size=(self.N, samples)))
        
    
    def sq_exp_kernel(a, b, l=1, sigma=1):
        'Uses an squared exponential kernel'
        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a,b.T)
        return sigma**2 * np.exp(sqdist/(-2*l))
    
    
    def posterior(self, x_test, kernel):
        # Compute the mean at test points
        n = len(x_test)
        Lk = np.linalg.solve(self.L, kernel(self.x, x_test))
        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.y))
        # Compute varience at test points
        K_ = kernel(x_test, x_test)
        std = np.sqrt(np.diag(K_) - np.sum(Lk**2, axis=0))
        #new_L = np.linalg.cholesky(K_ + self.noise_var * np.eye(n) - np.dot(Lk.T, Lk))
        #return mu.reshape(-1,1) + np.dot(new_L, np.random.normal(size=(n, self.samples)))
        return mu, std
    
def contour_len(x, y):
    for num, i in enumerate(x):
        x1 = x[num - 1]
        y1 = y[num - 1]
        x2 = x[num]
        y2 = y[num]

        try:
            hypotenuse_array.append(math.hypot((x1 - x2), (y1 - y2)))
        except NameError:
            hypotenuse_array = [math.hypot((x1 - x2), (y1 - y2))]
        return np.sum(np.array(hypotenuse_array))

def gaussian_process(spline_dict, kernel=GP.sq_exp_kernel, kernel_l=1, kernel_sigma=1, new_points=1000, plot_examples=False):
    for key in splines.keys(): # for each file
        im = io.imread(ts_path+'data/Processed/'+key+'_ZSensor_processed_grey.tif')
        for mol in splines[key]['Splines']: # for each molecule in file
            x = np.asarray(splines[key]['Splines'][mol]['x_coord']).reshape(-1,1)
            y = np.asarray(splines[key]['Splines'][mol]['y_coord']).reshape(-1,1)
            
            model = GP(x,y,kernel=kernel, kernel_l=kernel_l, kernel_sigma=kernel_sigma)
            f_prior = model.prior
            x_new = np.linspace(min(x),max(x), new_points)
            f_posterior = model.posterior(x_new, GP.sq_exp_kernel)

            if plot_examples:
                title='Gaussian Proccess Splining Results'
                x_tup, y_tup = (x.reshape(-1),x_new.reshape(-1)), (y.reshape(-1), f_posterior[0])
                labels = ('Topo', 'GP Splines')
                plot_overlays(x_tup, y_tup, im, title=title, labels=labels, std=f_posterior[1])

            print('GP: ', contour_len(x_new, f_posterior[0]))
            print('Topo: ', contour_len(x,y))
            print('|GP - Topo|: %.3f \n' %(contour_len(x_new, f_posterior[0])-contour_len(x,y)))

            # Overwrite current x and y coords
            splines[key]['Splines'][mol]['x_coord'] = x_new
            splines[key]['Splines'][mol]['y_coord'] = f_posterior[0]


#------- Run Script --------
ts_path = "/Users/Maxgamill/Desktop/Uni/PhD/TopoStats/"
splines = get_spline_jsons(ts_path + 'data/')
im = io.imread(ts_path+'data/Processed/20161024_339_LIN_6ng_434rep_8ng_PLL.026_ZSensor_processed_grey.tif')

gaussian_process(splines, new_points=1000, plot_examples=True)



