#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:30:13 2021

@author: Maxgamill
"""

import numpy as np
import imageio
import os
import json

'''
Module for importing images and creating more by rotations
 should also rotate and save segmentions (grains), 
 make a copy of the matching json with a correct name.
'''

def import_files(path, ext):
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

def spline_rotations(spline_dict, pixel_size):
    '''
    Want to pass in a spline dictionary, and create a temp dict
     which contains the new coords in a similar format:
         {rot_<XX>: {mol_num: {'y_coord':..., 'x_coord':...}}}
    '''
    
    spline_rot_dict = {'rot_90':{}, 'rot_180':{}, 'rot_270':{}}
    # rotation arays w.r.t. image rotations above (i.e. clockwise)
    #  with a translation to shift back to original axis

    for mol_num in spline_dict.keys():
        # original arrays
        y = np.asarray(spline_dict[mol_num]['y_coord'])
        x = np.asarray(spline_dict[mol_num]['x_coord'])
        
        # rot_90 gives x -> y, y -> -x
        spline_rot_dict['rot_270'][mol_num] = {'y_coord':[float(i) for i in (-x+pixel_size)],
                                              'x_coord':[float(i) for i in y]}
        # rot_180 gives x -> -x, y -> -y
        spline_rot_dict['rot_180'][mol_num] = {'y_coord':[float(i) for i in (-y+pixel_size)],
                                               'x_coord':[float(i) for i in (-x+pixel_size)]}        
        # rot_270 gives x -> -y, y -> x
        spline_rot_dict['rot_90'][mol_num] = {'y_coord':[float(i) for i in (x)],
                                               'x_coord':[float(i) for i in (-y+pixel_size)]}

    return spline_rot_dict

def add_spline_rotations(full_dict):
    '''
    Want to pass through the full dict and append to it new
     values which are the rotations.
    '''
    full_dict_cp = full_dict.copy()
    original_filenames = list(full_dict.keys())
    for fname in original_filenames:
        spline_rot_dict = spline_rotations(full_dict[fname]['Splines'], pixel_size=1024)
        param_dict = full_dict[fname].copy()
        
        full_dict_cp[fname+'_rot_90'] = param_dict.copy()
        full_dict_cp[fname+'_rot_180'] = param_dict.copy()
        full_dict_cp[fname+'_rot_270'] = param_dict.copy()
        
        full_dict_cp[fname+'_rot_90']['Splines'] = spline_rot_dict['rot_90']
        full_dict_cp[fname+'_rot_180']['Splines'] = spline_rot_dict['rot_180']
        full_dict_cp[fname+'_rot_270']['Splines'] = spline_rot_dict['rot_270']
    
    return full_dict_cp



ts_path = "/Users/Maxgamill/Desktop/Uni/PhD/TopoStats/data/"
new_path = "/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/"

img_dict = import_files(ts_path+'processed/', '.png')
grain_dict = import_files(ts_path, '.txt')

with open(new_path+"JSONs/434_PLL_REL_minicircles.json") as file:
    full_dict = json.load(file)

save_modified_array_files(img_dict, new_path+'Images/', '.png')
save_modified_array_files(grain_dict, new_path+'Segmentations/', '.txt')

with open(new_path+"JSONs/434_PLL_REL_minicircles_withrotations.json", 'w') as file:
    full_dict_cp = add_spline_rotations(full_dict)
    json.dump(full_dict_cp, file)


