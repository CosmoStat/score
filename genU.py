#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:15:23 2018

@author: fnammour
"""

import numpy as np

def makeU1(n,m):
    """Create a n x m numpy array with (i)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U1 = np.tile(np.arange(n),(m,1)).T
    return U1

def makeU2(n,m):
    """Create a n x m numpy array with (j)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U2 = np.tile(np.arange(m),(n,1))
    return U2

def makeU3(n,m):
    """Create a n x m numpy array with (1)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U3 = np.ones((n,m))
    return U3

def makeU4(n,m):
    """Create a n x m numpy array with (i^2+j^2)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U4 = np.add.outer(np.arange(n)**2,np.arange(m)**2)
    return U4

def makeU5(n,m):
    """Create a n x m numpy array with (i^2-j^2)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U5 = np.subtract.outer(np.arange(n)**2,np.arange(m)**2)
    return U5

def makeU6(n,m):
    """Create a n x m numpy array with (i*j)_{i,j} entries where i is the ith
    line and j is the jth column
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: n x m numpy array"""
    U6 = np.outer(np.arange(n),np.arange(m))
    return U6

def makeUi(n,m):
    """Create a 6 x n x m numpy array containing U1, U2, U3, U4, U5 and U6
    INPUT: n positive integer (number of lines),
           m positive (integer number of columns)
    OUTPUT: 6 x n x m numpy array"""
    return np.array([makeU1(n,m),makeU2(n,m),makeU3(n,m),makeU4(n,m),makeU5(n,m),makeU6(n,m)])

def G(X,U=None,W = 1):
    """Compute the 6 inner products of an image and U."""
    if U is None:
        U = makeUi(*X.shape)
    return np.array([(X*W*U[i]).sum() for i in range(6)])

def Moments(X, U=None, W=1, centered=True):
    """Compute all quadrupole moments of X up to order 2.
    Returns: (m_{00}, [x_c, y_c], (M_{ij})_{ij}),
    where: 
    - m_{00} is the 0th order moment, ie the (weighted) energy
    - [x_c,y_c] are the centroid (first order moments normalized by energy)
    - M is the 2x2 matrix containing all 2nd order moments, either centered
    or not.
    """
    GX = G(X,U,W)
    # uncentered moments
    Ms = [GX[2], np.array([GX[1],GX[0]]), np.empty((2,2))]
    Ms[1] /= Ms[0]
    Ms[2][0,0] = 1./2 * (GX[3]+GX[4]) 
    Ms[2][1,0] = GX[5] 
    Ms[2][0,1] = Ms[2][1,0]
    Ms[2][1,1] = 1./2 * (GX[3]-GX[4])
    
    if centered:
        # centered (2nd order) moments
        Ms[2][0,0] -= GX[0]**2/GX[2]
        Ms[2][1,0] -= GX[0]*GX[1] / GX[2]
        Ms[2][0,1] = Ms[2][1,0]
        Ms[2][1,1] -= GX[1]**2/GX[2]
    return Ms

def FindEll(X, U=None, W = 1):
    """Estimate the ellipticity parameters on an image."""
    Ms = Moments(X,U,W)
    mu = Ms[2]
    e1 = (mu[0,0]-mu[1,1])/(mu[0,0]+mu[1,1])
    e2 = 2*mu[0,1]/(mu[0,0]+mu[1,1])
    return np.array([e1,e2])

def MomentDeconvol(X, P, U=None):
    """Performs unweighted deconvolution in moment space, as
    in Melchior et al., 2011, Table 1, and returned ellipticities
    Note: here we are neglecting all first-order terms
    (i.e. we assume both X and P to be centered)"""
    M_im = Moments(X)
    M_psf = Moments(P)
    
    return im_cent - np.sum(X) * psf_cent
    








