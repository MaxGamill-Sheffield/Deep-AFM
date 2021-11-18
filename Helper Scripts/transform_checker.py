#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:43:18 2021

@author: Maxgamill
"""
import json
import numpy as np
import matplotlib.pyplot as plt

proj = '/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/'
json_file = '/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/JSONs/'

grains = np.loadtxt(proj + 'Segmentations/minicircle.spm_grains_rotate90.txt')
#grains = np.reshape(grains, [int(np.sqrt(grains.size)),int(np.sqrt(grains.size))])
with open(json_file + '434_PLL_REL_minicircles_withrotations.json') as file:
    full_dict = json.load(file)

def overlay(grains, splines_dict):
    plt.figure(figsize=(6.0,6.0))
    plt.title('Showing: All')

    grains_copy = np.zeros(grains.shape) + grains
    grains_copy[grains_copy!=0] = 255
    plt.imshow(grains_copy)
    
    for i in range(1,20):
        spline = splines_dict[str(i)]
        plt.plot(spline['x_coord'], spline['y_coord'],label=str(i))
        plt.legend(loc='best')
        plt.savefig('all_spline_overlay_for_all.png')
        


overlay(grains, full_dict['minicircle.spm_rot_90']['Splines'])    
    
    
    
    
    
    
    