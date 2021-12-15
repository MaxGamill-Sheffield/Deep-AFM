#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:03:55 2021

@author: Maxgamill
"""

import numpy as np
from skimage import io
import os


def import_files(path, ext):
    # Imports images from directpry into a dictionary
    filelist = os.listdir(path)
    file_dict = {}
    for file in filelist[:]:
        split_file = file.split('.')
        if ext == '.png' and split_file[-1] == 'png':
            file_dict[file.split(ext)[0]] = np.asarray(io.imread(path+file))
        elif ext == '.txt' and split_file[-1] == 'txt':
            contents = np.loadtxt(path+file)
            sq_array_len = np.sqrt(contents.size)
            file_dict[file.split(ext)[0]] = np.reshape(contents, 
                                                       [int(sq_array_len),
                                                        int(sq_array_len)])
        elif ext == '.ome.tiff' and split_file[-1] == 'tiff':
            file_dict[og_name_apeer(file).split(ext)[0]] = io.imread(path+file)
        else:
            pass
    return file_dict


def og_name_apeer(fname):
    # Converts apeer edited file names back to original format
    # uses static '5' from '..._<XXX/spm>_ZSensor_processed_grey_<class>'
    splits = fname.split('_')
    amended = '_'.join(splits[:len(splits)-5]) + '.' + '_'.join(splits[len(splits)-5:])
    return amended



def reorder_apeer(file_dict):
    full_dict = {}

    # Obtain image filenames and labels from grain file names
    for file in file_dict.keys():
        # Split full filename to get the label and image filename
        fname, label = file.rsplit('_',1)
        temp_dict = {}
        try:
            # if fname already present, add label and corresponding grain
            full_dict[fname][label] = file_dict[file]
        except:
            # if file not already present, make new label dict for full_dict
            temp_dict[label] = file_dict[file]
            full_dict[fname] = temp_dict
        
    return full_dict
        

def apeer_convert(full_dict):
    grain_dict = {}
    full_label_dict = {}
    
    for key in full_dict:
        label_dict = {}
        comb_array = np.asarray([full_dict[key][key2] for key2 in full_dict[key]])
        # obtain molecule numbers for each labeled grain
        labels = list(reversed(full_dict[key].keys()))
        mol_num = list(reversed(np.max(comb_array, axis=(1,2))))
        # get first array
        init = comb_array[0,:,:].copy() # gets the first label array
        # combine arrays
        for grain_idx in range(1,len(comb_array)):# loops over 1 to # of label arrays
            curr_array = comb_array[grain_idx,:,:] # gets label array
            curr_array_cp = curr_array.copy() # copies label array
            for mols_in_idx in range(mol_num[grain_idx],0,-1): # needs to be reversed otherwise indexes overlap
                # adds the mol number in index to the sum of mols in previous indexes to get a mol number in total image
                curr_array_cp[curr_array_cp==mols_in_idx]=sum(mol_num[:grain_idx])+mols_in_idx
            init += curr_array_cp
        # create label dict
        mol_count = sum(mol_num)
        for grain_idx in range(len(mol_num)):
            for mols_in_idx in range(mol_num[grain_idx],0,-1):
                label_dict[str(mol_count)] = labels[grain_idx]
                mol_count -= 1
            # compile dictionaries into main ones
            full_label_dict[key] = label_dict
            grain_dict[key] = init

    return grain_dict, full_label_dict


def remove_boundary_grains(grains):
    '''
    Finds unique label indicies in the grains file along the edges and
    removes them.

    Parameters
    ----------
    grains : ndarray
        An array formatted as per the readme.

    Returns
    -------
    grains : ndarray
        An array formatted as per the readme with any molecules residing
        on the edges removed.
    '''
    
    border_pixels = [grains[0,:], # first col
                     grains[:,0], # first row
                     grains[len(grains)-1,:], # last col
                     grains[:,len(grains)-1]] # last row
    for border in border_pixels:
        for mol_label in np.unique(border):       
            grains[grains==mol_label] = 0
    
    return grains









