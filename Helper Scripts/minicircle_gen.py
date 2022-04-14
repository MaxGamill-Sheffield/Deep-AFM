#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 07:33:20 2022

@author: Maxgamill
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


#set seed for consistent results 
np.random.seed(2021)

class Plots():
    def plot_xyrt(xyrt_array, square_size=1, save=False):
        'Plots an r vs t graph and the simulated minicircles'
        #setup subplots and axis
        fig, ax = plt.subplots(1,2, figsize=(11,5))
        ax[0].set_xlabel('t_radians')
        ax[0].set_ylabel('r(t)')
        ax[0].set_title('Radius of Minicircle')
        ax[1].set_xlabel('x(t)')
        ax[1].set_ylabel('y(t)')
        ax[1].set_title('Shape of Minicircle')
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
        if save:
            fig.savefig('Noise_comparison')
        return
    
    def plot_noise_comp(noise_array, sim_noise, save=False):
        'Plots sim and non-sim noises'
        fig, ax = plt.subplots(1,2, figsize=(10,6))
        pos1 = ax[0].imshow(noise_array, vmin=0, vmax=1.3)
        fig.colorbar(pos1, ax=ax[0])
        ax[0].set_title('Noise File')
        pos2 = ax[1].imshow(sim_noise, vmin=0, vmax=1.3)
        fig.colorbar(pos2, ax=ax[1])
        ax[1].set_title('Simulated Noise')
        plt.show()
        if save:
            fig.savefig('Noise_comparison')
        return
    
    def plot_fast_slow_noise(fast_noise_image, fast_nums, slow_noise_image, slow_nums, save=False):
        fig, ax = plt.subplots(1,2, figsize=(10,6))
        pos1 = ax[0].imshow(fast_noise_image, vmin=-0.2, vmax=0.6)
        fig.colorbar(pos1, ax=ax[0])
        ax[0].set_title('Fast Noise mu=%.3f, std=%.3f' %(fast_nums[0],fast_nums[1]))
        pos2 = ax[1].imshow(slow_noise_image, vmin=-0.2, vmax=0.6)
        fig.colorbar(pos2, ax=ax[1])
        ax[1].set_title('Slow Noise mu=%.3f, std=%.3f' %(slow_nums[0],slow_nums[1]))
        plt.show()
        if save:
            plt.savefig('Noise-Fast_vs_Slow')
        return
        
    def plot_circle_comp(gf_grid, tot_grid, real_noise_grid, save=False):
        'plots clean / sim noise / noise circles and slice distribution'
        fig, ax = plt.subplots(3,3, figsize=(15,11), tight_layout=True)
        fig.suptitle('Simulated Minicircle Noise Comparison: clean/simulated noise/real noise')
        ax[0,0].set_title('Image + Trace')
        ax[0,1].set_title('Image + Lines')
        ax[0,2].set_title('Line Trace Heights (nm)')
        
        ax[0,0].imshow(gf_grid, cmap='gray')
        ax[0,1].imshow(gf_grid, cmap='gray')
        ax[1,0].imshow(tot_grid, cmap='gray')
        ax[1,1].imshow(tot_grid, cmap='gray')
        ax[2,0].imshow(real_noise_grid, cmap='gray')
        ax[2,1].imshow(real_noise_grid, cmap='gray')
        
        for i in range(3):
            for j in range(len(xy_array)):
                ax[i,0].plot(xy_array[j,:,1],xy_array[j,:,0])
                ax[i,0].set_xlim(0,512)
                ax[i,0].set_ylim(512,0)
            ax[i,1].plot(np.linspace(10,500,3), [194,194,194])
            ax[i,1].plot([419,419,419], np.linspace(10,500,3), color='orange')
        ax[0,2].plot(gf_grid[194,:],label='Row heights')
        ax[0,2].plot(gf_grid[:,419], label='Column heights', color='orange')
        ax[1,2].plot(tot_grid[194,:],label='Row heights')
        ax[1,2].plot(tot_grid[:,419], label='Column heights', color='orange')
        ax[2,2].plot(real_noise_grid[194,:],label='Row heights')
        ax[2,2].plot(real_noise_grid[:,419], label='Column heights', color='orange')
        plt.show()
        if save:
            fig.savefig('simicircles_gaussian_filtered_noisy_comparison')
        return

def contour_length(x, y):
    'Calculates the contour length given by the x/y coords'
    # computes the length of the 2 coords on either end of the period
    cont_len = ((x[-1]-x[0])**2+(y[-1]-y[0])**2)**0.5
    for i in range(len(x)-1):
        # calcs euclidean distance between points and sums them all
        cont_len += ((x[i]-x[i+1])**2+(y[i]-y[i+1])**2)**0.5
    return cont_len

class Gen_mol():
    'Generates molecules of different shapes'
    def circle(H=15, size_limit=1, no_points=101, log_rng=(-0.5,-2.5)):
        # H is the number of circles you will sum to produce the distored circles
        # Randomize amplitude and phase
        amp = (np.multiply(np.random.rand(1,H)-0.5, np.logspace(log_rng[0],log_rng[1],H))).T
        phase = np.random.rand(1,H).T * 2*np.pi
        # Accumulate r(t) over 2*pi
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
    
    def fig8(H=15, size_limit=1, no_points=101, log_rng=(-0,-2)):
        # H is the number of circles you will sum to produce the distored circles
        # Randomize amplitude and phase
        amp = (np.multiply(np.random.rand(1,H)-0.5, np.logspace(log_rng[0],log_rng[1],H))).T
        phase = np.random.rand(1,H).T * 2*np.pi
        # Accumulate r(t) over 2*pi
        t = np.linspace(0,2*np.pi, no_points)
        r = np.ones(t.size)
        for h in range(H):
          r += amp[h]*np.sin(h*t+phase[h])
        # ensure smaller circles become roughly the same size as larger ones 
        r *= size_limit/max(r) + 0.1*size_limit*np.random.rand()
        # Reconstruct x(t), y(t)
        x = r * np.sin(t)
        y = r * np.sin(t)*np.cos(t)
        # roate randomly between 0 an pi
        theta = np.random.rand()*np.pi
        rot_x = x*np.cos(theta)+y*np.sin(theta)
        rot_y = y*np.cos(theta)-1*x*np.sin(theta)
        return np.asarray([rot_x,rot_y,r,t]).T

def arrange_circles(no_circle, square_size=512, H=10, size_limit=50, no_points=101, log_rng=(0,-1.5)):
    'Generates multiple distorted circles on a canvas'
    array = np.zeros((no_circle, no_points, 4)) # create empty array to add to later
    for i in range(no_circle):
        # generate a circle
        xyrt_array = Gen_mol.circle(H=H, size_limit=size_limit, no_points=no_points, log_rng=log_rng)
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

def apply_gaussian(skelly_grid, double=True):
    'Applies a gaussian filter to smooth out the skeletonised grid'
    std = 8
    if double: # apply gaussian x2 so have a larger segment thats then smoothed
        skelly_grid = gaussian_filter(skelly_grid, 1)
        skelly_grid[skelly_grid!=0] = 1
        std = 4
    return gaussian_filter(skelly_grid, std)

def gen_noise(image, noise_out_shape, plot=False, save_plot=False):
    'generates noise based on a gaussian with fast and slow axis std from an image'
    slow_mean = np.mean(np.mean(image, axis=0))/2 # mean of column means
    slow_std = np.mean(np.std(image, axis=0)) # std of column means
    fast_mean = np.mean(np.mean(image, axis=1)/2) # mean of row means
    fast_std = np.std(np.mean(image, axis=1)) # mean of stds of rows
    
    slow_noise_image = np.random.normal(slow_mean, slow_std, size=noise_out_shape)
    fast_noise_lines = np.random.normal(fast_mean, fast_std, size=(noise_out_shape[1]))
    fast_noise_image = np.resize(fast_noise_lines, (fast_noise_lines.size, noise_out_shape[0])).T
    tot_noise = fast_noise_image + slow_noise_image
    #tot_noise -= np.min(tot_noise) # shift until lowest val is 0
    if plot:
        Plots.plot_fast_slow_noise(fast_noise_image, (fast_mean,fast_std), slow_noise_image, (slow_mean,slow_std), save=save_plot)
    return tot_noise


array = arrange_circles(5, H=10, no_points=301, log_rng=(0, -1.5))

#a = create_rand_circle()


noise_image = '/Users/Maxgamill/Desktop/Uni/PhD/Project/Results/Simdata/noise.txt'
noise_array = np.loadtxt(noise_image)*1e9 # convert from nm

xy_array = array[:,:,0:2]
skelly = skeletonise(xy_array)
grid = gridify(skelly, 512)

gf_grid = apply_gaussian(grid)
gf_grid[gf_grid!=0] = gf_grid[gf_grid!=0] * 2 # DNA should be 2nm
sim_noise = gen_noise(noise_array, grid.shape)

tot_grid = gf_grid + sim_noise
real_noise_grid = gf_grid + noise_array

Plots.plot_circle_comp(gf_grid, tot_grid, real_noise_grid)


