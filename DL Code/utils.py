#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:15:21 2021

@author: Maxgamill
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class graphical():
    
    def show_img(image, spline_coords, grains):
        '''
        Shows an overlay of the image and splines, and grains and splines.

        Parameters
        ----------
        image : TYPE
            DESCRIPTION.
        splines_coords : TYPE
            DESCRIPTION.
        grains : numpy.ndarray
            An NxN array with pixel lables matching that of the README.

        Returns
        -------
        None.
        
        '''
        # Make a copy of grains as not to edit the original one
        grains_copy = np.zeros(grains.shape) + grains
        # Highlight all found grains
        grains_copy[grains_copy!=0] = 255
        
        ax1 = plt.subplot(121)
        ax1.plot(spline_coords[0], spline_coords[1]) #needs label
        plt.imshow(image)
        
        ax2 = plt.subplot(122)
        ax2.plot(spline_coords[0], spline_coords[1]) #needs label
        plt.imgshow(grains)
        
        plt.show()
        
class data():
    
    def load_grains(grains_path):
        filelist = os.listdir(grains_path)
        df = pd.DataFrame([], columns=('Image Name','Image Path','Grains'))
        '''
        for file in filelist:
            grain = np.loadtxt(grains_path+file)
            df.append(file.split('_grains')[0], grains_path+file)
        '''
        df = pd.concat([pd.DataFrame([[file.split('_grains')[0], grains_path+file]],
                                     columns=['Image Name','Image Path']) for file in filelist],
                       ignore_index=True)
        return df
        