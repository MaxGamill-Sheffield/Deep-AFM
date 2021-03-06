#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:15:21 2021

@author: Maxgamill
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import utils

class graphical():
    ''' Contains all graph producing functions'''
    
    def show_overlays(image, grain, spline_dict, single=0, save=0, axis=False):
        '''
        Takes an image, its backbones and segmentation and plots 2 or 3
            graphs - An image-spline graph, and a single/multi grain-spline 
            graph.

        Parameters
        ----------
        image : ndarray
            An image file loaded with scikit.io.
        spline_df : dict
            A dictionary formatted as the molecule number: {'x_coords': an 
            ndarray or list of x coords and the same for y coords.
        grain : ndarrray.
            A NxN numoy array with 0 as the background and integers related to
            the molecule number.
        single : int
            An integer label of the molecule number in the data. The default 
            is 0 meaning that a single molecule is not selected.
        save : str, optional
            A path to save the produced graph to. The default is 0 which
            means no file is saved.
        save : bool, optional
            A flag to show the axis on the graphs. (For judging scales)

        Returns
        -------
        None.

        '''
        
        tot_mols = len(spline_dict)
        
        if single==0:
            plot_no = 2
        else:
            assert isinstance(single,int) and single<=tot_mols and single>=0,\
            "`single` should be an integer <= %i." %tot_mols
            plot_no = 3
            grain_copy = np.zeros(grain.shape) + np.array(grain)
            grain_copy[grain_copy==single]=255
        
        # Create subplots for image and grain graphs
        fig, ax = plt.subplots(1,plot_no)
        ax[0].set_title('Image and Splines')
        ax[0].imshow(image, cmap='gray')
        ax[1].set_title('All Masks and Splines')
        ax[1].imshow(grain)
        
        # Plot the splines on the subplots
        for mol_num in spline_dict.keys():
            ax[0].plot(spline_dict[mol_num]['x_coord'],
                       spline_dict[mol_num]['y_coord'],label = mol_num)
            ax[1].plot(spline_dict[mol_num]['x_coord'],
                       spline_dict[mol_num]['y_coord'],label = mol_num)
            # Plot single grain and spline graph
            if single!=0 and mol_num==str(single):
                ax[2].set_title('Mask %i and Spline' %(single))
                ax[2].imshow(grain_copy)
                ax[2].plot(spline_dict[mol_num]['x_coord'],
                           spline_dict[mol_num]['y_coord'],label = mol_num)
                
        # Show legend
        ax[0].legend(loc='center right', bbox_to_anchor=(0, 0.5))
        if axis==0:
            for i in ax:
                i.set_axis_off()
        
        # Save the figure to the path specified
        if save != 0:
            plt.savefig(save)
        
    def show_batch(sample_batched):
        ''' UNFINISHED
        Should be made to do show overlays but iterate through a batch on
        one figure
        '''
        images_batch, grains_batch = sample_batched['Image'], sample_batched['Grain']
        batch_size = len(images_batch) # needed when adding splines
        im_size = images_batch.size(2) # needed when adding splines
        grid_border_size=2 # needed when adding splines
        
        #grid = utils.make_grid(images_batch)
        #plt.imshow(grid.numpy().transpose((1, 2, 0)))
        #plt.title('Batch from dataloader')
        #plt.show()
        
        # get figure layout
        if batch_size <= 2 or batch_size % 2 == 1:
            rows=batch_size
            cols=1
            fig, ax = plt.subplots(int(rows), 2*cols)
            for i in range(int(rows)):
                ax[i, 0].imshow(images_batch[i].numpy().transpose((1,2,0)))
                ax[i, 0].set_axis_off()
                ax[i, 1].imshow(grains_batch[i].numpy().transpose((1,2,0)))
                ax[i, 1].set_axis_off()
        else:
            rows=batch_size/2
            cols=2
            # As plotting image and grain, need dims x 2
            fig, ax = plt.subplots(int(rows), 2*cols)
            for i in range(int(rows)):
                ax[i, 0].imshow(images_batch[i].numpy().transpose((1,2,0)))
                ax[i, 0].set_axis_off()
                ax[i, 1].imshow(grains_batch[i].numpy().transpose((1,2,0)))
                ax[i, 1].set_axis_off()
                ax[i, 2].imshow(images_batch[i+int(rows)].numpy().transpose((1,2,0)))
                ax[i, 2].set_axis_off()
                ax[i, 3].imshow(grains_batch[i+int(rows)].numpy().transpose((1,2,0)))
                ax[i, 3].set_axis_off()
        fig.title('')        
        plt.axis('off')
        plt.show()

        
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
            
    def get_labels(json_dict, key):
        '''
        May or may not need this as dataframes don't work well with dataloader
        '''
        class_labels = json_dict[key]['Circular']
        contours = json_dict[key]['Contour Lengths']
        df_cols = ['Molecule Number', 'Circular', 'Contour Lengths']
        df = pd.concat([pd.DataFrame([[mol_num,
                                             class_labels[mol_num],
                                             contours[mol_num]]],
                                           columns = df_cols) for mol_num in json_dict[key]['Circular'].keys()])
        
        return df
            
            
            
            
            
            
            
            
        