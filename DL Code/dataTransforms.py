#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:21:45 2021

@author: Maxgamill
"""

from skimage import transform
import torch
import numpy as np

class Rescale(object):
    ''' Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    '''
            
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        # get data from sample dict
        image, grain = \
            sample['Image'], sample['Grain'] #, sample['Splines'], sample['Data']
        # get height and width from the image shape
        h, w = image.shape[:2] # Although only 1 channel, may have more in future
        
        if isinstance(self.output_size, int):
            # if output_size is an int, calulate new h/w based on h:w ratio
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            # if output_size is a tuple, set it to the output_size
            new_h, new_w = self.output_size
        
        # ensure pixel sizes are integers
        new_h, new_w = int(new_h), int(new_w)
        # rescale image and grains without splining / anti-aliasing the grains
        img = transform.resize(image, (new_h, new_w))
        grains = transform.resize(grain, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False)
        '''
        # rescale splines
        for mol_num in splines.keys():
            splines[mol_num]['x_coord'] = splines[mol_num]['x_coord'] * new_h / h
            splines[mol_num]['y_coord'] = splines[mol_num]['y_coord'] * new_w / w
        '''
        sample = {'Image': img, 'Grain': grains} # 'Splines': splines, 'Data': data}
        
        return sample
            
            
class ToTensor(object):
    ' Converts np arrays to tensors UNFINISHED - need to include splines & data'
    # get data from sample dict
    def __call__(self, sample):
        # Extract sample parts
        image, grain = sample['Image'], sample['Grain'] #, sample['Splines'], sample['Data']
        
        # numpy image: H x W x C (as np axis 0,1,2) & torch image: C x H x W (as np axis 2,0,1)
        # -> Will have to add channel input/selection too when using multiple channels.
        image = np.expand_dims(image, axis=0)
        grain = np.expand_dims(grain, axis=0)

        '''
        for mol_num in splines.keys():
            splines[mol_num]['x_coord'] = torch.tensor(splines[mol_num]['x_coord'])
            splines[mol_num]['y_coord'] = torch.tensor(splines[mol_num]['y_coord'])
        '''
        # May or may not need to convert other labels to tensors
        
        sample = {'Image': torch.from_numpy(image),
                   'Grain': torch.from_numpy(grain)} #, 'Splines': splines, 'Data': data
        
        return sample
    
class NormaliseTiff(object):
    'Sets TIFF image values between 0 and 1 by dividing by 65535'
    def __call__(self, sample):
        # Extract sample parts
        image, grain = sample['Image'], sample['Grain']
        # Divide by max TIFF pixel value
        image = image/65535
        # Reform sample
        sample = {'Image': image,
                   'Grain': grain}
        return sample
    
        