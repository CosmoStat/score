#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:04:42 2020

@author: fnammour
"""

from score import score
import numpy as np
import pickle

#define paths
root_path = '/Users/fnammour/Documents/Thesis/direct_deconvolution/'
data_path = root_path+'Data/300_sersic/'
results_path = root_path+'Implementation/results/150_conv_k4/'

#load data
#set SNRs and load the SNR-gamma correspondence dictionary
SNRs = [40,75,150,380]
pickle_in = open(results_path+'SNR_gamma.pkl','rb')
SNR_gamma = pickle.load(pickle_in)

#Load ground truth galaxies
gals = np.load(data_path+'galaxies.npy')
PSF = np.load(data_path+'PSFs_10000.npy')
gal_num,row,column = gals.shape

#set denoising parameters
n_starlet = 4 #number of starlet scales
n_shearlet = 3 #number of shearlet scales
lip_eps = 1e-3 #error upperbound for Lipschitz constant
tolerance = 1e-6 #to test convergence
n_itr = 150 #number of iteration
k = 4 #Set k for k-sigma hard thresholding
beta_factor = 0.95 #to ensure that beta is not too big
rip = True #Removal of Isolated Pixel in the deconvolution solution

#instantiate the solver
solver = score(k=k,n_starlet=n_starlet,n_shearlet=n_shearlet,epsilon=lip_eps,\
               rip=rip,tolerance=tolerance,beta_factor=beta_factor,\
               verbose=False)

#loop on SNR
for SNR in SNRs:
    #set gamma according to the SNR
    gamma = SNR_gamma[SNR]
    #loop on the galaxy images
    for GT,Y,H in zip(gals,noisy_gals,PSF):
        None
    print(SNR)
    