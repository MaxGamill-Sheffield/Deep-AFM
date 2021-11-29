#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:47:23 2021

@author: Maxgamill
"""

import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import utils
import os.path
from skimage import io


class dataLoaderSegmentation(Dataset):
    
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
        self.img_params = utils.df_it.get_img_params(self.data_dict)
        # set image and grain paths
        self.img_path = images_path
        self.grains_path = grains_path
        self.transform = transform
        
    def __len__(self):
        return len(self.img_params)
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image
        img_name = os.path.join(self.img_path, self.img_params.iloc[idx, 1])
        image = io.imread(img_name)
        
        # Get grain
        grain_name = os.path.join(self.grains_path, self.img_params.iloc[idx, 2])
        grain = np.loadtxt(grain_name) 
        
        # Get spline
        key = self.img_params.iloc[idx, 0]
        spline_df = utils.df_it.get_splines(self.data_dict[key]['Splines'])
    
        # Compile into dictionary as a "sample"
        sample = {'Image': image, 'Grain': grain, 'Splines': spline_df}
        
        # Compute transofrms
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
    
path = "/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/"

data = dataLoaderSegmentation(images_path=str(path+'Images/'), json_path=str(path+'JSONs/relaxed_minicircles.json'), grains_path=str(path+'/Segmentations/'))
