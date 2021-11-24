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


def save_modified_array_files(array_dict, path_out, ftype):
    for fname in array_dict.keys():
        
        array = array_dict[fname]
        array_rot90 = rotate(array)
        array_rot180 = rotate(array_rot90)
        array_rot270 = rotate(array_rot180)
        
        array_flip_vertical = np.flip(array,axis=0)
        array_flip_horizontal = np.flip(array,axis=1)
        array_rot90_flip_vertical = np.flip(array_rot90,axis=0)
        array_rot90_flip_horizontal = np.flip(array_rot90,axis=1)
        
        if ftype == '.png':
            # <name+transform>_<channel>_<colour>.png
            label = fname.split('_',len(fname.split('_'))-3)[-1]
            name = fname.split('_'+label)[0]
            imageio.imsave(path_out+fname+'.png', array)
            imageio.imsave(path_out+name+'-rot-90_'+label+'.png', array_rot90)
            imageio.imsave(path_out+name+'-rot-180_'+label+'.png', array_rot180)
            imageio.imsave(path_out+name+'-rot-270_'+label+'.png', array_rot270)
            
            imageio.imsave(path_out+name+'-flip-horizontal_'+label+'.png', array_flip_horizontal)
            imageio.imsave(path_out+name+'-flip-vertical_'+label+'.png', array_flip_vertical)
            imageio.imsave(path_out+name+'-rot-90-flip-horizontal_'+label+'.png', array_rot90_flip_horizontal)
            imageio.imsave(path_out+name+'-rot-90-flip-vertical_'+label+'.png', array_rot90_flip_vertical)
        elif ftype == '.txt':
            name = fname[:-7] # <name>
            np.savetxt(path_out+fname+'.txt', array)
            np.savetxt(path_out+name+'-rot-90_grains.txt', array_rot90)
            np.savetxt(path_out+name+'-rot-180_grains.txt', array_rot180)
            np.savetxt(path_out+name+'-rot-270_grains.txt', array_rot270)
            
            np.savetxt(path_out+name+'-flip-horizontal_grains.txt', array_flip_horizontal)
            np.savetxt(path_out+name+'-flip-vertical_grains.txt', array_flip_vertical)
            np.savetxt(path_out+name+'-rot-90-flip-horizontal_grains.txt', array_rot90_flip_horizontal)
            np.savetxt(path_out+name+'-rot-90-flip-vertical_grains.txt', array_rot90_flip_vertical)
        else:
            print("Only png and txt files can be made at this time: ", fname)

def spline_transformations(spline_dict, pixel_size):
    '''
    Want to pass in a spline dictionary, and create a temp dict
     which contains the new coords in a similar format:
         {rot_<XX>: {mol_num: {'y_coord':..., 'x_coord':...}}}
    '''
    
    spline_trans_dict = {'rot_90':{}, 'rot_180':{}, 'rot_270':{},
                       'rot_90_vert':{}, 'rot_90_hori':{},
                       'og_vert':{}, 'og_hori':{}}
    # rotation arays w.r.t. image rotations above (i.e. clockwise)
    #  with a translation to shift back to original axis

    for mol_num in spline_dict.keys():
        # original arrays
        y = np.asarray(spline_dict[mol_num]['y_coord'])
        x = np.asarray(spline_dict[mol_num]['x_coord'])
        
        # rot_90 gives x -> y, y -> -x
        spline_trans_dict['rot_270'][mol_num] = {'y_coord':[float(i) for i in (-x+pixel_size)],
                                              'x_coord':[float(i) for i in y]}
        # rot_180 gives x -> -x, y -> -y
        spline_trans_dict['rot_180'][mol_num] = {'y_coord':[float(i) for i in (-y+pixel_size)],
                                               'x_coord':[float(i) for i in (-x+pixel_size)]}        
        # rot_270 gives x -> -y, y -> x
        spline_trans_dict['rot_90'][mol_num] = {'y_coord':[float(i) for i in (x)],
                                               'x_coord':[float(i) for i in (-y+pixel_size)]}
        
        # flip_vert gives y -> -y
        spline_trans_dict['og_vert'][mol_num] = {'y_coord':[float(i) for i in (-y+pixel_size)],
                                               'x_coord':[float(i) for i in (x)]}
        # flip_hori gives x -> -x
        spline_trans_dict['og_hori'][mol_num] = {'y_coord':[float(i) for i in (y)],
                                               'x_coord':[float(i) for i in (-x+pixel_size)]}
        
        spline_trans_dict['rot_90_vert'][mol_num] = {'y_coord':[float(i) for i in (-x+pixel_size)],
                                                     'x_coord':[float(i) for i in (-y+pixel_size)]}
        spline_trans_dict['rot_90_hori'][mol_num] = {'y_coord':[float(i) for i in (x)],
                                                     'x_coord':[float(i) for i in (y)]}
        
    return spline_trans_dict

def add_spline_rotations(full_dict):
    '''
    Want to pass through the full dict and append to it new
     values which are the rotations.
    '''
    full_dict_cp = full_dict.copy()
    original_filenames = list(full_dict.keys())
    for fname in original_filenames:
        spline_trans_dict = spline_transformations(full_dict[fname]['Splines'], full_dict[fname]['Image Parameters']['x_px'])
        
        param_dict = full_dict[fname].copy()
        
        full_dict_cp[fname+'-rot-90'] = param_dict.copy()
        full_dict_cp[fname+'-rot-180'] = param_dict.copy()
        full_dict_cp[fname+'-rot-270'] = param_dict.copy()
        
        full_dict_cp[fname+'-flip-vertical'] = param_dict.copy()
        full_dict_cp[fname+'-flip-horizontal'] = param_dict.copy()
        full_dict_cp[fname+'-rot-90-flip-horizontal'] = param_dict.copy()
        full_dict_cp[fname+'-rot-90-flip-vertical'] = param_dict.copy()
        
        full_dict_cp[fname+'-rot-90']['Splines'] = spline_trans_dict['rot_90']
        full_dict_cp[fname+'-rot-180']['Splines'] = spline_trans_dict['rot_180']
        full_dict_cp[fname+'-rot-270']['Splines'] = spline_trans_dict['rot_270']
    
        full_dict_cp[fname+'-flip-vertical']['Splines'] = spline_trans_dict['og_vert']
        full_dict_cp[fname+'-flip-horizontal']['Splines'] = spline_trans_dict['og_hori']
        full_dict_cp[fname+'-rot-90-flip-vertical']['Splines'] = spline_trans_dict['rot_90_vert']
        full_dict_cp[fname+'-rot-90-flip-horizontal']['Splines'] = spline_trans_dict['rot_90_hori']
    
    return full_dict_cp



ts_path = "/Users/Maxgamill/Desktop/Uni/PhD/TopoStats/data/"
new_path = "/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/"

img_dict = import_files(ts_path+'Processed/', '.png')
grain_dict = import_files(ts_path, '.txt')

with open(new_path+"JSONs/test.json") as file:
    full_dict = json.load(file)

save_modified_array_files(img_dict, new_path+'Images/', '.png')
save_modified_array_files(grain_dict, new_path+'Segmentations/', '.txt')

with open(new_path+"JSONs/test_transforms.json", 'w') as file:
    full_dict_cp = add_spline_rotations(full_dict)
    json.dump(full_dict_cp, file)


