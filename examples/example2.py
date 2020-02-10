#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  10 10:28:03 2020

@author: fnammour
"""

# =============================================================================
#      THIS CODE IS AN EXAMPLE OF A USE CASE OF SCORE IN DECONVOLUTION
# =============================================================================


from score import score
import numpy as np

# %% load data
#true galaxies
gals = np.load('true_galaxies.npy')
#observed galaxies
gals_obs = np.load('observed_galaxies.npy')
#Point Spread Function
psfs = np.load('psfs.npy')

# %% perform deconvolution of the chosen galaxy image for gamma = 0 and 1
#initiate two instances of score
#set the value of gamma
g1 = score(gamma=1,verbose=False)
g0 = score(gamma=0,verbose=False,rip=False)
#initiate lists of ellipticity relative errors
g1_error_list = list()
g0_error_list = list()

#loop
for obs, psf, gt in zip(gals_obs,psfs,gals):
    #deconvolve
    g1.deconvolve(obs=obs,ground_truth=gt,psf=psf)
    g0.deconvolve(obs=obs,ground_truth=gt,psf=psf)
    #update ellipticity error lists
    g1_error_list += [g1.relative_ell_error]
    g0_error_list += [g0.relative_ell_error]

g1_error = np.array(g1_error_list)
g0_error = np.array(g0_error_list)
# %% show results
print('Mean Ellipticity Error for gamma=1: {}'.format(g1_error.mean()))
print('Mean Ellipticity Error for gamma=0: {}'.format(g0_error.mean()))