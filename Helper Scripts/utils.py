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
        filename, file_ext = file.split('.', 1)
        if ext == '.png' and file_ext == 'png':
            file_dict[file.split(ext)[0]] = np.asarray(io.imread(path+file))
        elif ext == '.txt' and file_ext == 'txt':
            contents = np.loadtxt(path+file)
            sq_array_len = np.sqrt(contents.size)
            file_dict[file.split(ext)[0]] = np.reshape(contents, 
                                                       [int(sq_array_len),
                                                        int(sq_array_len)])
        elif ext == '.ome.tiff' and file_ext == 'ome.tiff':
            file_dict[filename] = io.imread(path+file)
        else:
            pass
    return file_dict


def reorder_apeer(file_dict):
    full_dict = {}
    uniq_fnames = []
    uniq_labels = []

    for file in file_dict.keys():
        fname, label = file.rsplit('_',1)
        uniq_fnames.append(fname)
        uniq_labels.append(label)
        
    uniq_fnames = set(uniq_fnames)
    uniq_labels = set(uniq_labels)
        
    for fname in uniq_fnames:
        temp_dict = {}
        for label in uniq_labels:
            temp_dict[label] = file_dict[fname+'_'+label]
        full_dict[fname] = temp_dict
        
    return full_dict
        
def apeer_convert(full_dict):
    grain_dict = {}
    full_label_dict = {}
    
    for key in full_dict:
        label_dict = {}
        
        comb_array = np.asarray([full_dict[key][key2] for key2 in full_dict[key]])
        #obtain molecule numbers for each labeled grain
        labels = list(full_dict[key].keys())
        mol_num = np.max(comb_array, axis=(1,2))
        # initalise first array
        init = comb_array[0,:,:].copy()
        # combine arrays
        for grain_idx in range(1,len(comb_array)):
            curr_array = comb_array[grain_idx,:,:]
            curr_array_cp = curr_array.copy()
            curr_array[curr_array!=0] = 1
            init += curr_array + curr_array_cp*mol_num[grain_idx-1]
        # create label dict
        mol_count = 1
        for grain_idx in range(len(mol_num)):
            for mols_in_idx in range(1,mol_num[grain_idx]+1):
                label_dict[str(mol_count)] = labels[grain_idx]
                mol_count += 1
        
        full_label_dict[key] = label_dict
        grain_dict[key] = init


    return grain_dict, full_label_dict







