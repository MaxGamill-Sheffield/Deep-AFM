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
import utils as uv


class dataLoaderSegmentation(Dataset):
    
    def __init__(self, images_path, json_path, grains_path):
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
        super(dataLoaderSegmentation, self).__init__()
        # load json file
        with open(json_path) as file:
            self.data_dict = json.load(file)
        # load txt files
        self.img_path = images_path
        self.grains = uv.data.load_grains(grains_path)
        
    def __len__(self):
        return len(self.data_dict)
        
    def __getitem__(self, idx):
        '''
            if torch.is_tensor(idx):
                idx = idx.tolist()
        
            img_name = os.path.join(self.root_dir,
                                    self.landmarks_frame.iloc[idx, 0])
            image = io.imread(img_name)
            landmarks = self.landmarks_frame.iloc[idx, 1:]
            landmarks = np.array([landmarks])
            landmarks = landmarks.astype('float').reshape(-1, 2)
            sample = {'image': image, 'landmarks': landmarks}
        
            if self.transform:
                sample = self.transform(sample)
        '''
        return idx
        
    
path = "/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/"

data = dataLoaderSegmentation(images_path=str(path+'Images/'), json_path=str(path+'JSONs/relaxed_minicircles.json'), grains_path=str(path+'/Segmentations/'))
