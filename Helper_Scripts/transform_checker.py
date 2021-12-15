#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:43:18 2021

@author: Maxgamill
"""
import json
import numpy as np
import matplotlib.pyplot as plt


def overlay(grains, splines_dict, title=None):
    ''' Plots grains overlayed with splines'''
    
    plt.figure(figsize=(6.0,6.0))

    grains_copy = np.zeros(grains.shape) + grains
    grains_copy[grains_copy!=0] = 255 # Highlight grains
    plt.imshow(grains_copy)
    
    for key in splines_dict.keys():
        spline = splines_dict[key]
        plt.plot(spline['x_coord'], spline['y_coord'],label=key)
        plt.legend(loc='best')
        plt.axis('off')
        plt.title(title)
        #plt.savefig('all_spline_overlay_for_all.png')
 
    
    
    
    
    
    
    