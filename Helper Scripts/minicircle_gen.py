#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 07:33:20 2022

@author: Maxgamill
"""

import numpy as np
import matplotlib.pyplot as plt


#set seed for consistent results 
np.random.seed(2021)

def contour_length(x, y):
    # computes the length of the 2 coords on either end of the period
    end_len = ((x[-1]-x[0])**2+(y[-1]-y[0])**2)**0.5
    cont_len=0
    for i in range(len(x)-1):
        # calcs euclidean distance between points
        cont_len += ((x[i]-x[i+1])**2+(y[i]-y[i+1])**2)**0.5
    cont_len += end_len
    return cont_len


def create_rand_circle(H=15, size_limit=1, no_points=101, log_rng=(-0.5,-2.5)):
    # H is the number of circles you will sum to produce the distored circles
    # Randomize amplitude and phase
    amp = (np.multiply(np.random.rand(1,H)-0.5, np.logspace(log_rng[0],log_rng[1],H))).T
    phase = np.random.rand(1,H).T * 2*np.pi
    # Accumulate r(t) over t=[0,2*pi]
    t = np.linspace(0,2*np.pi, no_points)
    r = np.ones(t.size)
    for h in range(H):
      r += amp[h]*np.sin(h*t+phase[h])
    # ensure smaller circles become roughly the same size as larger ones 
    r *= size_limit/max(r) + 0.1*size_limit*np.random.rand()
    # Reconstruct x(t), y(t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.asarray([x,y,r,t]).T

def arrange_circles(no_circle, square_size=512, H=10, size_limit=50, points=101, log_rng=(0,-1.5)):
    'Generates multiple distorted circles on a canvas'
    array = np.zeros((no_circle, points, 4)) # create empty array to add to later
    for i in range(no_circle):
        # generate a circle
        xyrt_array = create_rand_circle(H=H, size_limit=size_limit, points=points, log_rng=log_rng)
        # adds a bias to the new random centres
        rand_centre_x = np.random.randint(-10,square_size+10) 
        rand_centre_y = np.random.randint(-10,square_size+10)
        xyrt_array[:,0] += rand_centre_x
        xyrt_array[:,1] += rand_centre_y
        print('Mol %i contour length: %.3f' %(i, contour_length(xyrt_array[:,0], xyrt_array[:,1])))
        # put coords into array
        array[i] += xyrt_array
    plt.show()
    return array

def skeletonise(xy_array):
    'Maps coords to grid'
    xy_copy = xy_array.copy()
    xy_copy[0] = np.round(xy_copy[0])
    xy_copy[0] = np.round(xy_copy[0])
    return xy_copy

def gridify(xy_array, grid_size, grid_ext=10):
    'Turn the skeletonised array to a numpy grid'
    arr_max = int(np.max(xy_array))+1
    grid = np.zeros((arr_max, arr_max))
    for mol_xy in xy_array:
        for idx in range(len(mol_xy[:,0])):
            grid[int(mol_xy[idx,0]),int(mol_xy[idx,1])] = 1
    return grid[0:grid_size,0:grid_size]

def plot_xyrt(xyrt_array, square_size=1):
    'Plots an r vs t graph and the simulated minicircles'
    #setup subplots and axis
    fig, ax = plt.subplots(1,2, figsize=(11,5))
    ax[0].set_xlabel('t_radians')
    ax[0].set_ylabel('r(t)')
    ax[0].set_title('Cartesian Periodic Functions (Minicircles)')
    ax[1].set_xlabel('x(t)')
    ax[1].set_ylabel('y(t)')
    ax[1].set_title('Polar Periodic Functions')
    if len(xyrt_array.shape) == 3: # plot multiple circles
        ax[1].set_ylim(bottom=0,top=square_size)
        ax[1].set_xlim(xmin=0,xmax=square_size)
        for i in range(xyrt_array.shape[0]):
            ax[0].plot(xyrt_array[i,:,3], xyrt_array[i,:,2]) # plot r vs t
            ax[1].plot(xyrt_array[i,:,0], xyrt_array[i,:,1]) # plot minicircles
            #ax[1].scatter(rand_centre_x, rand_centre_y) there must be a way to find the shift coords
    else: # plot single circle
        ax[0].plot(xyrt_array[:,3], xyrt_array[:,2])
        ax[1].scatter(0,0, label='centre')
        ax[1].plot(np.cos(xyrt_array[:,3]), np.sin(xyrt_array[:,3]), 'k--', label='circle')
        ax[1].plot(xyrt_array[:,0], xyrt_array[:,1], label = 'distorted circle')
        ax[1].legend(loc='upper right')
    plt.show()
    return


array = arrange_circles(10, H=10, points=101, log_rng=(0, -1.5))

#a = create_rand_circle(plot=True)

xy_array = array[:,:,0:2]
skelly = skeletonise(xy_array)
grid = gridify(skelly, 512)
plt.imshow(grid)

'''
x,y,t,r = gen_circ()

fig, ax = plt.subplots(1,2, figsize=(11,5))
ax[0].plot(t, r)
ax[0].set_xlabel('t_radians')
ax[0].set_ylabel('r(t)')
ax[0].set_title('Cartesian Periodic Function')
ax[1].scatter(0,0, label='centre')
ax[1].plot(np.cos(t), np.sin(t), 'k--', label='circle')
ax[1].plot(x, y, label = 'distorted circle')
ax[1].set_xlabel('x(t)')
ax[1].set_ylabel('y(t)')
ax[1].set_title('Polar Periodic Function')
ax[1].legend(loc='upper right')
'''