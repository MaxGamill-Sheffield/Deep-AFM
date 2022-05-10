#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:47:27 2022

@author: Maxgamill
"""

import pandas as pd
import numpy as np

class CompareSetup():    
    def pairings(test_df, truth_df, threshold=5e-9):
        '''From two ts dataframes, finds grain number from each which show
            the lowest error in the grain center, below a specified threshold.
        '''
        pair_list = []
        for i in truth_df.index:
            # get test and truth differences for x and y centres
            x_diff = abs(test_df['grain_center_x']-truth_df['grain_center_x'][i])
            y_diff = abs(test_df['grain_center_y']-truth_df['grain_center_y'][i])
            # sum x and y errors and find the minimum value's index
            error_df = x_diff + y_diff
            if min(error_df) < threshold:
                idx_pair = ((error_df).idxmin(), i) # get dataframe index-wise pair
                pair_list.append([test_df['grain no'][idx_pair[0]],truth_df['grain no'][idx_pair[1]]])
        return pd.DataFrame(pair_list, columns=['test_grain_no','truth_grain_no'])
    
    def compare(test_df, truth_df, pairs, tracestats=False):
        '''Takes 2 dataframes and best mol pairings, and produces a dataframe
            containing the percentage difference in the analyses.'''
        cols = list(truth_df.columns) + ['test_grain_no', 'truth_grain_no']
        diffs = pd.DataFrame(columns=cols) # makes empty dataframe with columns
        # obtain the correct molecule number label by dropping str cols
        if tracestats:
            mol_col = 'Molecule number'
            name_col = 'Image Name'
            test_df['Circular'] = test_df['Circular']*1+1 # turns bools into T=2 or F=1 for calc below
            truth_df['Circular'] = truth_df['Circular']*1+1 # calc produces 0 and 0.5 when matching not
        else:
            mol_col = 'grain no'
            name_col = 'filename'
            grain_no_col = 'grain no'
        # start looping over data entries for each set of pairs
        for i in range(len(pairs)):
            pair = pairs.loc[i] # gets the 2 grain numbers in the pair
            filename = truth_df[name_col][truth_df[mol_col]==pair[1]] # gets the filname associated with the pair
            temp_truth = truth_df[truth_df[mol_col]==pair[1]].drop(name_col, axis=1) # saves the dropped columns
            temp_test = test_df[test_df[mol_col]==pair[0]].drop(name_col, axis=1) # saves the dropped columns
            temp_truth = temp_truth.drop(grain_no_col, axis=1) # saves the dropped columns
            temp_test = temp_test.drop(grain_no_col, axis=1)

            # calculate the percentage difference for each value in the row
            diffs_percents = (temp_test-temp_truth.values) / abs(temp_truth.values)
            diffs_percents = pd.DataFrame(diffs_percents) # convert from series to df
            # obtain and re-insert the corresponding grain number and filenames for each entry
            l = int(diffs_percents.shape[1])
            diffs_percents.insert(l, 'test_grain_no', pair[0])
            diffs_percents.insert(l+1, 'truth_grain_no', pair[1])
            diffs_percents.insert(0, name_col, filename)
            if tracestats:
                # turns circular calc outputs to 1=no match, 0=match
                diffs_percents['Circular'] = diffs_percents['Circular'].replace([0.5],1)
            diffs = pd.concat([diffs, diffs_percents], ignore_index=True)
        return diffs
            
    def diffs(test_df, truth_df, tr_test=None, tr_truth=None):
        'Computes the percentage difference for an entire TopoStats.csv dataset'
        # the tracesats files must go in here too as this is where the pairs are made
        cols = list(truth_df.columns) + ['test_grain_no','truth_grain_no']
        all_diffs = pd.DataFrame(columns=cols)
        for fname in test_df['filename'].unique():
            test_df_sieved = test_df.loc[test_df.iloc[:,0] == fname]
            truth_df_sieved = truth_df.loc[truth_df.iloc[:,0] == fname]
            pairs = CompareSetup.pairings(test_df_sieved, truth_df_sieved)
            diffs = CompareSetup.compare(test_df_sieved, truth_df_sieved, pairs)
            all_diffs = pd.concat([all_diffs, diffs], ignore_index=True)
        all_diffs = all_diffs.drop(['grain no'], axis=1)
        # check if just ts file present or ts and tr and return appropriately
        if type(tr_test)==pd.core.frame.DataFrame and type(tr_truth)==pd.core.frame.DataFrame:
            cols = list(tr_truth.columns) + ['test_grain_no','truth_grain_no']
            all_diffs_tr = pd.DataFrame(columns=cols)
            for fname in tr_test['Image Name'].unique():
                tr_test_sieved = tr_test.loc[tr_test.iloc[:,0] == fname]
                tr_truth_sieved = tr_truth.loc[tr_truth.iloc[:,0] == fname]
                pairs = pairs-1
                diffs_tr = CompareSetup.compare(tr_test_sieved, tr_truth_sieved, pairs, tracestats=True)
                all_diffs_tr = pd.concat([all_diffs_tr, diffs_tr], ignore_index=True)
            all_diffs_tr = all_diffs_tr.drop(['Molecule number'], axis=1)
            return all_diffs, all_diffs_tr
        return all_diffs






# import topostats data
new = pd.read_csv("/Users/Maxgamill/Desktop/Uni/PhD/TopoStats_removegwy/TopoStats/topostats/data/processed_minicircle/grain_stats/grainstats.csv")
old = pd.read_csv("/Users/Maxgamill/Desktop/Uni/PhD/TopoStats_pull/TopoStats/data/data.csv")
# find like columns
new_cols = list(new.columns)
old_cols = list(old.columns)

shared=[]
for i in old_cols:
    for j in new_cols:
        if i==j:
            shared.append(i)

#trim the columns of original data
new_trim = new[shared[1:]]
old_trim = old[shared[1:]]

# compute differences
ts_differences = CompareSetup.diffs(new_trim, old_trim)
# save output dataFrame
ts_differences.to_csv("/Users/Maxgamill/Desktop/Uni/PhD/TopoStats_removegwy/comp2.csv")
