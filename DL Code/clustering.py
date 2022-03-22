#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:08:56 2022

@author: Maxgamill
"""

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def get_grain_img(sample):
    directory = sample['directory']
    filename = sample['filename']
    grain_no = sample['grain no']
    string = directory + '/' + filename + '.spm_grains.npy'
    mask = np.load('/Users/Maxgamill/Desktop/Uni/PhD/TopoStats_pull/TopoStats/data/'+str(string))
    mask[mask!=grain_no] = 0
    string = directory + '/Processed/' + filename + '.spm0_Height_processed.tiff'
    img = io.imread('/Users/Maxgamill/Desktop/Uni/PhD/TopoStats_pull/TopoStats/data/'+str(string))
    return mask, img

def get_bb(mask):
    'Returns bouunding box indicies of image'
    arr = mask.nonzero()
    horr = (min(arr[0]),max(arr[0]))
    vert = (min(arr[1]),max(arr[1]))
    return horr, vert

def get_mol(df, cluster, examples=4):
    files = df[df['Cluster']==cluster].iloc[:,1:4] # get files and grain no's
    sample = files.sample(examples, ignore_index=True)
    fig, ax = plt.subplots(examples,2, tight_layout=True, figsize=(8,12))
    for i in range(examples):
        mask, img = get_grain_img(sample.loc[i])
        horr, vert = get_bb(mask)
        fig.suptitle('Mask and Image Samples of Cluster %i (%i molecules)' %(cluster, len(files)))
        ax[i,0].imshow(mask[horr[0]:horr[1], vert[0]:vert[1]])
        ax[i,1].imshow(img[horr[0]:horr[1], vert[0]:vert[1]])
    return fig

def evaluate(clustering):
    label_list = list(clustering.labels_)
    no_classes = len(set(clustering.labels_))
    val = 0
    for i in set(label_list):
        if i == -1:
            val += abs(0-label_list.count(i))
        else:
            base = len(clustering.labels_)/(no_classes-1)
            val += abs(base-label_list.count(i))/(no_classes-1)
    # find way to really evaluate func
    return val - 40*no_classes

def test(eps_lst, min_samp_lst, data):
    if isinstance(eps_lst, (float,int)) or isinstance(min_samp_lst, (float,int)):
        clustering = DBSCAN(eps=eps_lst, min_samples=min_samp_lst).fit(data)
        label_list = list(clustering.labels_)
        for i in set(label_list):
            print('Cluster %i has %i objects' %(i, label_list.count(i)))
    else:
        df = np.zeros((len(eps_lst),len(min_samp_lst)))
        for i in range(len(eps_lst)):
            for j in range(len(min_samp_lst)):
                clustering = DBSCAN(eps=eps_lst[i], min_samples=min_samp_lst[j]).fit(data)
                df[i][j] += evaluate(clustering)
        df = pd.DataFrame(df,index=eps_lst,columns=min_samp_lst)
        return df, (df.min(axis=1).idxmin(), df.min().idxmin()) #idx is min_samp, eps

topo = pd.read_csv('/Users/Maxgamill/Desktop/Uni/PhD/Project/Results/Clustering/data.csv')
data = topo.iloc[:,4:24] # get numerical data
data = data.drop(['grain_center_x','grain_center_y'], axis=1)#,'grain_maximum','grain_mean','grain_median'], axis=1) #remove spatially dependant fields
#normalised_data = (data-data.mean())/data.std()
standard_data = StandardScaler().fit_transform(data) # normalise data

'''
eps_lst = np.linspace(1,5,101)
min_samp_lst = list(range(4,41))
df, idxs = test(eps_lst, min_samp_lst, standard_data)
print(idxs)
test(idxs[0],idxs[1],standard_data)
'''


eps = 1.52
min_samp = 28
clustering = DBSCAN(eps=eps, min_samples=min_samp).fit(standard_data)

label_list = list(clustering.labels_)
for i in set(label_list):
    print('Cluster %i has %i objects' %(i, label_list.count(i)))
topo['Cluster'] = label_list


cluster = 2
fig = get_mol(topo, cluster, examples=5)
save_string = '/Users/Maxgamill/Desktop/Uni/PhD/Project/Results/Clustering/'+str(eps)+'eps_'+str(min_samp)+'minsamples_cluster'+str(cluster)+'.png'
plt.savefig(save_string)



