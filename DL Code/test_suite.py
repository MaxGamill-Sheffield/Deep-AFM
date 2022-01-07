#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:10:54 2022

@author: Maxgamill
"""

'''
This is the testing suite for all modules within the DL Code directory
'''
import pytest
import torch
import numpy as np
import pandas as pd
import dataloader
import dataTransforms


class Test_dataloader:
    'Tests for dataTransforms.py file functions'
    @pytest.fixture(scope='class', autouse='True') # Uses the return 
    def dataset(self):
        # Builds a randomised image and grain to create a sample
        images_path = 'example_data/Images/'
        json_path = 'example_data/JSONs/relaxed_minicircles.json'
        grains_path = 'example_data/Segmentations/'
        return dataloader.SegmentationData(images_path, json_path, grains_path)
        
    def test_notEmpty(self, dataset):
        assert len(dataset) > 0 # check if dataset contains items
        # check that all parts of a sample aren't empty
        assert isinstance(dataset[0]['Image'], np.ndarray)
        assert isinstance(dataset[0]['Grain'], np.ndarray)
        assert dataset[0]['Splines']
        assert isinstance(dataset[0]['Data'], pd.core.frame.DataFrame)
        

class Test_dataTransforms:
    'Tests for dataTransforms.py file functions'
    @pytest.fixture(scope='class', autouse='True') # Uses the return 
    def sample(self):
        # Builds a randomised image and grain to create a sample
        image_size = (1024,1024)
        image = np.random.randint(0,65535,image_size)
        grain = np.random.randint(0,10,image_size)
        return {'Image': image, 'Grain': grain}
    
    def test_Rescale(self, sample, rescale=512):
        rescaled_samples = dataTransforms.Rescale(rescale)(sample)
        # Checks that rescaled shapes match the recale value
        assert rescaled_samples['Image'].shape == (rescale, rescale) #check shape is correct
        assert rescaled_samples['Grain'].shape == (rescale, rescale) #check shape is correct
        
    def test_ToTensor(self, sample):
        tensored_samples = dataTransforms.ToTensor()(sample)
        # Checks outputs are torch.Tensor types
        assert type(tensored_samples['Image']) == torch.Tensor #check is a tensor
        assert type(tensored_samples['Grain']) == torch.Tensor #check is a tensor
        
    def test_NormaliseTiff(self, sample):
        normalised_samples = dataTransforms.NormaliseTiff()(sample)
        # Checks that image values are in the range 0-1
        assert np.sum(normalised_samples['Image'] > 1) == 0 #check if vals > 1
        assert np.sum(normalised_samples['Image'] < 0) == 0 #check if vals < 0
    
    