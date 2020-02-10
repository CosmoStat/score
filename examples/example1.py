#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:58:03 2020

@author: fnammour
"""

# =============================================================================
#        THIS CODE IS AN EXAMPLE OF A USE CASE OF SCORE IN DENOISING
# =============================================================================


from score import score
import numpy as np
import matplotlib.pyplot as plt

# %% load data
#true galaxies
gals = np.load('convolved_galaxies.npy')
#observed galaxies
gals_obs = np.load('observed_galaxies.npy')

#pick a galaxy image number (between 0 and 4, in this case)
gal_num = 2
gal = gals[gal_num]
gal_obs = gals_obs[gal_num]

# %% perform denoising of the chosen galaxy image
#initiate score
#set the value of gamma for example
denoiser = score(gamma=0.5)
#denoise
denoiser.denoise(obs=gal_obs) #the result will be in denoiser.solution

# %% plot result
vmin = np.min(gal_obs)
vmax = np.max([gal_obs,gal,denoiser.solution])

plt.figure(figsize=(15,15))

plt.subplot(311)
plt.imshow(gal,vmin=vmin,vmax=vmax,cmap='gist_stern')
plt.colorbar()
plt.title('True Galaxy')

plt.subplot(312)
plt.imshow(gal_obs,vmin=vmin,vmax=vmax,cmap='gist_stern')
plt.colorbar()
plt.title('Observed Galaxy')

plt.subplot(313)
plt.imshow(denoiser.solution,vmin=vmin,vmax=vmax,cmap='gist_stern')
plt.colorbar()
plt.title('Reconstructed Galaxy')

plt.suptitle('Restoration Results of Galaxy #{}'.format(gal_num))

plt.show()