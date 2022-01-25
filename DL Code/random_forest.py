#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:47:50 2022

@author: Maxgamill
"""

import numpy as np
import json
import os
from skimage import io, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import matplotlib.pyplot as plt


class RandomForest():
    
    def __init__(self, training_image, training_mask, sigma_min, sigma_max):
        'Initialises the classifier and feature function'
        self.features_func = partial(feature.multiscale_basic_features,\
                                intensity=True, edges=True, texture=True,
                                sigma_min=sigma_min, sigma_max=sigma_max)
        
        features = self.features_func(training_image)
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=10,\
                                     max_samples=0.05)
        self.clf = future.fit_segmenter(training_mask, features, clf)

    def output(self, new_image):
        'Uses the initialised classifier to predict features of the new image'
        features_new = self.features_func(new_image)
        return future.predict_segmenter(features_new, self.clf)
        

def show_output(img, result):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(13.5, 4))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Image for segmentation')
    ax[1].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
    ax[1].contour(result)
    ax[1].set_title('Image, mask and segmentation boundaries')
    ax[2].imshow(result)
    ax[2].set_title('Segmentation')
    fig.tight_layout()



#-------Running Script-------
train_img = io.imread('example_data/Random Forest/base_img.tiff', as_gray=True)
train_mask = io.imread('example_data/Random Forest/full.tiff')
#training_labels = io.imread('example_data/Random Forest/open_circle.tiff')
test_img = io.imread('example_data/Random Forest/test_img1.tiff', as_gray=True)

train_mask[train_mask!=0] = 1 # label all molecules as 1
train_mask[train_mask==0] = 2 # label all background as 2

forest = RandomForest(train_img, train_mask, 1, 2)
result = forest.output(test_img)

show_output(test_img, result)


