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
import maxUtils as mtils


path = "/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/"
#ts = None
ts = (transforms.Compose([dataTransforms.Rescale(512),dataTransforms.ToTensor()]))

# Load the data and transform it via rescaling so all are the same

trans_dataset = dataloader.SegmentationData(images_path=str(path+'Images/'),
                                   json_path=str(path+'JSONs/relaxed_minicircles.json'),
                                   grains_path=str(path+'/Segmentations/'),
                                   transform=ts)


# Setup iterator 
dataload = DataLoader(trans_dataset, batch_size=4, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataload):
    print(i_batch, sample_batched['Image'].size(), sample_batched['Grain'].size())
    # observe 4th batch and stop.
    if i_batch == 3:
        break


