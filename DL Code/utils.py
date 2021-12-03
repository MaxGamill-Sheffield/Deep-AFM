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
from torchvision import utils

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
        
    def show_batch(sample_batched):
        
        images_batch, grains_batch, spline_batch = \
            sample_batched['Image'], sample_batched['Grain'], sample_batched['Splines']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        
        fig, ax = plt.subplots(2,1)
        
        grid = utils.make_grid(images_batch)
        ax[0].imshow(grid.numpy().transpose((1, 2, 0)))
        grid2 = utils.make_grid(grains_batch)
        ax[1].imshow(grid2.numpy().transpose((1, 2, 0)))

        for i in range(spline_batch):
            ### needs adjusting for xy
            plt.scatter(spline_batch[i, :, 0].numpy() + i * im_size,
                    spline_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')
        


            plt.title('Batch from dataloader')
        
        
        
        plt.title('Batch from dataloader')
        
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
            

            
            
            
            
            
            
            
            
            
        