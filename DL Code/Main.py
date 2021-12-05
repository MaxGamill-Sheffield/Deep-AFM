#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:37:41 2021

@author: Maxgamill
"""

import dataloader
import dataTransforms
from torchvision import transforms
from torch.utils.data import DataLoader


path = "/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/"

# Load the data and transform it via rescaling so all are the same
'''
trans_dataset = dataloader.SegmentationData(images_path=str(path+'Images/'),
                                   json_path=str(path+'JSONs/relaxed_minicircles.json'),
                                   grains_path=str(path+'/Segmentations/'),
                                   transform=(transforms.Compose([dataTransforms.Rescale(512),dataTransforms.ToTensor()])))
'''
dataset = dataloader.SegmentationData(images_path=str(path+'Images/'),
                                   json_path=str(path+'JSONs/relaxed_minicircles.json'),
                                   grains_path=str(path+'/Segmentations/'),
                                   transform=(transforms.Compose([dataTransforms.Rescale(256)])))

# Setup iterator 
#dataload = DataLoader(trans_dataset)




