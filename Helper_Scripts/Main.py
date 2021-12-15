#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:37:21 2021

@author: Maxgamill
"""

import json
import utils
import image_transforms as it
import json_edditor as je


# Set paths for TopoStats, TopoStats produced JSON, resultant image data and
#apeer if using it
ts_path = "/Users/Maxgamill/Desktop/Uni/PhD/TopoStats/"
json_path = "/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/JSONs/"
data_path = "/Users/Maxgamill/Desktop/Uni/PhD/Project/Data/"
tracestats = je.read_json(ts_path + 'tracestats.json')
splines = je.get_spline_jsons(ts_path + 'data/')
merged_json = je.merge_data(tracestats, splines)


# If using Apeer annotated masks data
'''
apeer_path = "/Users/Maxgamill/Desktop/Uni/PhD/Project/Apeer/"
apeer_dict = utils.import_files(apeer_path,'.ome.tiff')
apeer_dict = utils.reorder_apeer(apeer_dict)
grain_dict, label_dict = utils.apeer_convert(apeer_dict)
'''
# If using sole TopoStats data
grain_dict = utils.import_files(ts_path+'data/', '.txt')

# Import images
img_dict = utils.import_files(ts_path+'data/Processed/', '.png')

# Modify and save images, and grains
it.save_modified_array_files(img_dict, data_path+'Images/', '.png')
it.save_modified_array_files(grain_dict, data_path+'Segmentations/', '.txt')

# Load JSON containing splines, perform transforms, and save them to the same 
#file
with open(json_path+"relaxed_minicircles.json", 'w') as file:
    full_dict_cp = it.add_spline_translations(merged_json)
    json.dump(full_dict_cp, file)
    
    
    
    
    
    
    
    
    
    