import numpy as np
from sf_tools.image.shape import Ellipticity
import galsim
from galsim.hsm import FindAdaptiveMom

def EllipticalGaussian(e1, e2, sig, xc=None, yc=None, stamp_size=(256,256)):
    if xc is None:
        xc = stamp_size[0]/2
    if yc is None:
        yc = stamp_size[0]/2
    # compute centered grid
    ranges = np.array([np.arange(i) for i in stamp_size])
    x = np.outer(ranges[0] - xc, np.ones(stamp_size[1]))
    y = np.outer(np.ones(stamp_size[0]),ranges[1] - yc)
    # shift it to match centroid
    xx = (1-e1/2)*x - e2/2*y
    yy = (1+e1/2)*y - e2/2*x
    # compute elliptical gaussian
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))

def GenNoise(SNR, obj):
    s = np.mean(np.linalg.norm(obj,axis=(1,2))**2)
    sigsq = s/(SNR*obj.shape[1]**2)
    return np.random.normal(scale=np.sqrt(sigsq),size=obj.shape)

def AdaptativeGaussian(obj, rmOutliers = True):
    stamp_size = obj[0,:,:].shape
    
    # convert to galsim images
    galsim_obj = [galsim.Image(itm) for itm in obj]
    
    # compute adaptive moments 
    all_moms = [FindAdaptiveMom(itm, strict=False) for itm in galsim_obj]
    # Check HSM flags and create mask to remove outliers
    flags = np.array([mom.moments_status for mom in all_moms])
    mask = (flags+1).astype('bool')
    if rmOutliers:
        temp = []
        for idx,val in enumerate(mask):
            if val:
                temp += [all_moms[idx]]        
        all_moms = temp
    Ws = np.array([EllipticalGaussian(-1.*mom.observed_shape.e1, mom.observed_shape.e2, 
                                      #convention fix: e1 sign swap
                                      mom.moments_sigma,
                                      mom.moments_centroid.y-1, mom.moments_centroid.x-1,
                                      # convention fix: swap x and y and origin at (0,0)
                                      stamp_size) for mom in
                                      all_moms])
    return Ws,mask
