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
import json

class graphical():
    ''' Contains all graph producing functions'''
    
    def show_overlays(image, grain, spline_df, single=0, save=0):
        '''
        Takes an image, its backbones and segmentation and plots 2 or 3
            graphs - An image-spline graph, and a single/multi grain-spline 
            graph.

        Parameters
        ----------
        image : ndarray
            An image file loaded with scikit.io.
        spline_df : pandas DataFrame
            A pandas DataFrame with columns of the molecule number, an array
            of x coords and and array of y coords.
        grain : ndarrray.
            A NxN numoy array with 0 as the background and integers related to
            the molecule number.
        single : int
            An integer label of the molecule number in the data. The default 
            is 0 meaning that a single molecule is not selected.
        save : str, optional
            A path to save the produced graph to. The default is 0 which
            means no file is saved.

        Returns
        -------
        None.

        '''
        tot_mols = len(spline_df)
        
        if single==0:
            plot_no = 2
        else:
            assert isinstance(single,int) and single<=tot_mols and single>=0,\
            "`single` should be an integer <= %i." %tot_mols
            plot_no = 3
            grain_copy = np.zeros(grain.shape) + grain
            grain_copy[grain_copy==single]=255
        
        # Create subplots for image and grain graphs
        fig, ax = plt.subplots(1,plot_no)
        ax[0].set_title('Image and Splines')
        ax[0].imshow(image, cmap='gray')
        ax[0].axis('off')
        ax[1].set_title('All Masks and Splines')
        ax[1].imshow(grain)
        ax[1].axis('off')
        
        # Plot the splines on the subplots
        for mol_num in range(tot_mols):
            ax[0].plot(spline_df['x'][mol_num], spline_df['y'][mol_num],
                     label = int(mol_num)+1)
            ax[1].plot(spline_df['x'][mol_num], spline_df['y'][mol_num],
                     label = int(mol_num)+1)
            # Plot single grain and spline graph
            if single!=0 and mol_num==single-1:
                ax[2].set_title('Mask %i and Spline' %(single))
                ax[2].imshow(grain_copy)
                ax[2].plot(spline_df['x'][mol_num], spline_df['y'][mol_num],
                         label = int(mol_num)+1)
                ax[2].axis('off')
                
        # Show legend
        ax[0].legend(loc='center right', bbox_to_anchor=(0, 0.5))
        
        # Save the figure to the path specified
        if save != 0:
            plt.savefig(save)
            
        
class df_it():
    '''Converts data to more accessable and usable dataframes'''
    
    def get_img_params(json_dict):
        '''
        Takes the dictionary produced by the helper script and turns the
            image information into a pandas dataframe. Can use obj.head() to
            check the columns and examples.

        Parameters
        ----------
        json_dict : dict
            A python dictionary produced by the helper scripts. Formatted 
            according to the README.

        Returns
        -------
        df : pandas DataFrame
            Formatted by columns of the dictionary keys, the matching image
            and grain files, and x/y pixel/length information.

        '''
        
        df_cols = ['Name','Image Name','Grain Name','x_px','y_px','x_real','y_real']
        df = pd.concat([pd.DataFrame([[keys,keys+'_ZSensor_processed_grey.png',keys+'_grains.txt',
                                      json_dict[keys]['Image Parameters']['x_px'],
                                      json_dict[keys]['Image Parameters']['y_px'],
                                      json_dict[keys]['Image Parameters']['x_len'],
                                      json_dict[keys]['Image Parameters']['y_len']]],
                                     columns=df_cols) for keys in json_dict.keys()], ignore_index=True)
        return df
            
    
    def get_splines(spline_dict):
        '''
        Gets splines created from the helper script dictionary and puts them
            into a dataframe for easy access.

        Parameters
        ----------
        spline_dict : dict
            The spline dictionary created from the helper script.

        Returns
        -------
        df : pandas DataFrame
            Formatted by molecule number, x coordinates and y coordinates.

        '''
        df_cols = ['Molecule Number','x','y']
        df = pd.concat([pd.DataFrame([[mol_num,
                                      np.asarray(spline_dict[mol_num]['x_coord']),
                                      np.asarray(spline_dict[mol_num]['y_coord'])]],
                                     columns=df_cols) for mol_num in spline_dict.keys()], ignore_index=True)
        return df
            
            
            
            
            
            
            
            
            
        