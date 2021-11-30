#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:21:45 2021

@author: Maxgamill
"""

class Rescale(object):
    ''' Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    '''
    
    def __init__(self, output_size):
        '''
        Initialises the instance by getting the desired output size.

        Parameters
        ----------
        output_size : int or tuple
            Desired output size.

        Returns
        -------
        None.

        '''
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
    def __call__(self, sample):
        
        
        
        
        
        
        
        
        
        