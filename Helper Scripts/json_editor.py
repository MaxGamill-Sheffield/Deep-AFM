#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:41:09 2021

@author: Maxgamill
"""
import json
import os


def get_spline_jsons(path):
    # Make empty dictionary for storing splines
    spline_dict = {}
    # Gets all files in a directory
    filelist = os.listdir(path)
    # Reduces filelist to only those ending with ".json"
    for file in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(file.endswith('.json')):
            filelist.remove(file)
        else:
            name = file.split('_spline')[0]
            data = read_json(path + file)
            spline_dict[name] = data
    return spline_dict


def read_json(path):
    # Read tracestats.json file
    with open(path) as file:
        return json.load(file)


def merge_data(tracestats, splines):
    new_dict = {}
    for filename in splines.keys():
        temp_dict = {'Splines': {},
                     'Circular': {},
                     'Contour Lengths': {},
                     'Image Parameters': {}
                     }
        file_splines = splines[filename]
        temp_dict['Splines'] = file_splines['Splines']
        temp_dict['Image Parameters'] = file_splines['Img_params']
        
        for i in range(len(tracestats['Image Name'])):
            if tracestats['Image Name'][str(i)] == filename:
                mol_num = tracestats['Molecule number'][str(i)]
                temp_dict['Circular'][str(mol_num+1)] = tracestats['Circular'][str(i)]
                temp_dict['Contour Lengths'][str(mol_num+1)] = tracestats['Contour Lengths'][str(i)]
            else:
                pass
        new_dict[filename] = temp_dict
        
    return new_dict
