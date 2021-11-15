#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:30:13 2021

@author: Maxgamill
"""

import numpy as np
import imageio
import os

'''
Module for importing images and creating more by rotations
 should also rotate and save segmentions (grains), 
 make a copy of the matching json with a correct name.
'''

def import_img(path, ext):
    # Imports images from directpry into a dictionary
    filelist = os.listdir(path)
    file_dict = {}
    for file in filelist[:]:
        file_ext = file.split('.')[-1]
        if ext == '.png' and file_ext == 'png':
            file_dict[file.split(ext)[0]] = np.asarray(imageio.imread(path+file))
        elif ext == '.txt' and file_ext == 'txt':
            contents = np.loadtxt(path+file)
            sq_array_len = np.sqrt(contents.size)
            file_dict[file.split(ext)[0]] = np.reshape(contents, 
                                                       [int(sq_array_len),
                                                        int(sq_array_len)])
        else:
            pass
    return file_dict


def rotate(img_array):
    # rotates 90 deg clockwise
    return np.flip(img_array.T, axis=1)


def save_modified_array_files(array_dict, path, ext):
    for key in array_dict.keys():
        array = array_dict[key]
        array_rot90 = rotate(array)
        array_rot180 = rotate(array_rot90)
        array_rot270 = rotate(array_rot180)
        
        if ext == '.png':
            imageio.imsave(path+key+'.png', array)
            imageio.imsave(path+key+'_rotate90.png', array_rot90)
            imageio.imsave(path+key+'_rotate180.png', array_rot180)
            imageio.imsave(path+key+'_rotate270.png', array_rot270)
        elif ext == '.txt':
            np.savetxt(path+key+'.txt', array)
            np.savetxt(path+key+'_rotate90.txt', array_rot90)
            np.savetxt(path+key+'_rotate180.txt', array_rot180)
            np.savetxt(path+key+'_rotate270.txt', array_rot270)
        else:
            print("Only png and txt files can be made at this time")


ts_path = "/Users/Maxgamill/Desktop/Uni/PhD/TopoStats/data/"
new_path = "/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/"

img_dict = import_img(ts_path+'Processed/', '.png')
grain_dict = import_img(ts_path, '.txt')

#save_modified_array_files(img_dict, new_path+'Images/', '.png')
save_modified_array_files(grain_dict, new_path+'Segmentations/', '.txt')







