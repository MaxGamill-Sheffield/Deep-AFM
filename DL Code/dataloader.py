#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:47:23 2021

@author: Maxgamill
"""

import torch
import json
import numpy as np
from torch.utils.data import Dataset
import maxUtils as mtils
import os.path
from skimage import io
import pandas as pd


class SegmentationData(Dataset):
    
    def __init__(self, images_path, json_path, grains_path, transform=None):
        '''
        Initialises the dataloader class by compiling the image, json and grain
         data together.

        Parameters
        ----------
        images_path : str
            Path to the location of all the image files.
        json_path : str
            Path to JSON formalised as in the README file.
        grains_path : str
            NxN Numpy array containing labeled pixels as in the README.

        Returns
        -------
        None.

        '''

        # load json file
        with open(json_path) as file:
            self.data_dict = json.load(file)
        # set get image parameter dataframe
        self.img_params = mtils.df_it.get_img_params(self.data_dict)
        # set image and grain paths
        self.img_path = images_path
        self.grains_path = grains_path
        self.transform = transform
        
    def __len__(self):
        ''' Gets the length of the dataset '''
        return len(self.img_params)
        
    def __getitem__(self, idx):
        '''
        Accepts an index and returns labels and data for that index of the 
            data loader.

        Parameters
        ----------
        idx : int
            The index number of the data in the dataset.

        Returns
        -------
        sample : dict
            A sample of the image, grain for that image and splines of the
                molecules in that image.

        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image
        img_name = os.path.join(self.img_path, self.img_params.iloc[idx, 1])
        image = io.imread(img_name).astype('float64')
        
        # Get grain
        grain_name = os.path.join(self.grains_path, self.img_params.iloc[idx, 2])
        grain = np.loadtxt(grain_name, dtype='int8')
        
        # Get spline
        key = self.img_params.iloc[idx, 0]
        spline_dict = self.data_dict[key]['Splines']
        for mol_num in spline_dict:
            spline_dict[mol_num]['x_coord'] = np.asarray(spline_dict[mol_num]['x_coord'])
            spline_dict[mol_num]['y_coord'] = np.asarray(spline_dict[mol_num]['y_coord'])
            
        # Get contour lengths and class labels in pandas df
        data = mtils.df_it.get_labels(self.data_dict, key)
        
        # Compile into dictionary as a "sample"
        sample = {'Image': image, 'Grain': grain, 'Splines': spline_dict, 'Data': data}
        
        # Compute transforms
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    