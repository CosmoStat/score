#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@authors: fnammour, Morgan A. Schmitz
"""
import numpy as np
from modopt.signal.wavelet import get_mr_filters, filter_convolve
from genU import makeUi

def hard_thresh(signal, threshold):
    """Apply hard thresholding."""
    return signal*(np.abs(signal)>=threshold)

def sigma_mad(signal):
    """Estimate sigma."""
    return 1.4826*np.median(np.abs(signal-np.median(signal)))

def MS_hard_thresh(wave_coef, n_sigma):
    """Apply mutliscale hard thresholding."""
    wave_coef_rec_MS = np.zeros(wave_coef.shape)
    for i,wave in enumerate(wave_coef):
        wave_coef_rec_MS[i,:,:] = hard_thresh(wave, n_sigma[i])
    return wave_coef_rec_MS

def norm2(signal):
    """Compute l2 norm of a signal."""
    return np.linalg.norm(signal,2)

def norm1(signal):
    """Compute l1 norm of a signal."""
    return np.linalg.norm(signal,1)

def G(X,U,W = 1):
    """Compute the 6 inner products of an image and U."""
    return np.array([(X*W*U[i]).sum() for i in range(6)])

def prior(alpha):
    """Compute the sparsity constraint of the loss."""
    norm = 0
    for wave in alpha:
        norm += norm1(wave)
    return norm

def comp_mu(W,U):
    """Compute the normalization coefficients mu of the moments contraint."""
    return np.array([1/(6*norm2(W*U[i])**2) for i in range(6)])

def comp_thresh(alpha,k=4):
    """Compute the threshold for HT."""
    thresholds = []
    thresholds += [(k+1)*sigma_mad(alpha[0])]
    for wave in alpha[1:-1]:
        thresholds += [k*sigma_mad(wave)]
    return np.array(thresholds)

def reconstruct(alpha, positivity = True):
    """Inverse of starlet transform."""
    X = alpha.sum(0)
    if positivity:
        X = X*(X>0)
    return X

def FindEll(X, U, W = 1):
    """Estimate the ellipticity parameters on an image."""
    GX = G(X,U,W)
    mu20 = 0.5*(GX[3]+GX[4])-GX[0]**2/GX[2]
    mu02 = 0.5*(GX[3]-GX[4])-GX[1]**2/GX[2]
    mu11 = GX[5]-GX[0]*GX[1]/GX[2]
    e1 = (mu20-mu02)/(mu20+mu02)
    e2 = 2*(mu11)/(mu20+mu02)
    return np.array([e1,e2])

class Cadmos(object):
    def __init__(self,Y,gamma=7,k=5,W=None,X0=None,GT=None,loss=[],nb_updates=5,
                 past_iterations=0):
        self.Y = np.copy(Y)
        self.gamma = gamma
        self.k = k
        self.loss = loss
        self.nb_updates = nb_updates
        if self.nb_updates:
            self.update_thresh = True
        self.nit = past_iterations

        row,column = self.Y.shape # careful - untested with rectangular images

        # Gaussian windows
        if W is None:
            self.W = np.ones(Y.shape)
        else:
            self.W = W

        # (Optional) ground truth for diagnostics
        self.GT = GT

        # Get wavelet filters
        self.filters = get_mr_filters((row,column), coarse = True)

        # Starlet transform observations
        self.alphaY = filter_convolve(self.Y, self.filters) #Starlet transform of Y

        #Calculate thresholds using Y, the observed galaxy
        self.thresholds = comp_thresh(self.alphaY,self.k)

        # First guess
        if X0 is None:
            # if no first guess, initialize with uniform image
            self.X = np.ones(Y.shape) / Y.size
        else:
            self.X = np.copy(X0)
        self.alpha = filter_convolve(self.X, self.filters) #Starlet transform of X

        # U matrices 
        self.U = makeUi(row, column)

        #Moment constraints normalization
        self.mu = comp_mu(self.W,self.U)

        #Lipschitz constant of the gradient
        L = 1 + self.gamma*np.array([
                self.mu[i]*norm2(self.W*self.U[i])**2 
                for i in range(6)]).sum()
        self.beta = 1./L
        
    def comp_loss(self, X, alpha):
        """Compute loss."""
        R = X - self.Y
        mom_cons = G(R,self.U,self.W)
        this_loss = [norm2(R)**2/2., (self.gamma*mom_cons**2/2.).sum(), prior(alpha)]
        self.loss += [this_loss]

    def comp_grad(self, X):
        """ Compute the loss gradient."""
        R = X - self.Y
        mom_cons = G(R,self.U,self.W)
        return self.gamma*self.W*np.array(
               [self.mu[i]*mom_cons[i]*self.U[i] 
                for i in range(6)]).sum(0) + R


    def update_thresholds(self):
        R = self.Y - self.X
        alphaR = filter_convolve(R, self.filters)
        self.thresholds = comp_thresh(alphaR,self.k) 

    def ForwardBackwardIter(self):
        #Fidelity gradient according to alpha
        grad = self.comp_grad(self.X)
        
        #Update X
        self.X -= self.beta*grad

        #Starlet transform of X
        self.alpha = filter_convolve(self.X, self.filters)
        
        #Compute LOSS for the first half of the iteration
        self.comp_loss(self.X, self.alpha)

        #Update thresholds
        if self.update_thresh:
            if self.nit < self.nb_updates:
                self.update_thresholds()
            else:
                self.update_thresh = False
        
        #Multiscale threshold except coarse scale
        self.alpha[:-1] = MS_hard_thresh(self.alpha[:-1], self.beta*np.array(self.thresholds))
        
        #Reconstruct X
        self.X = reconstruct(self.alpha)
        
        #Calculate loss for the second half of the iteration
        self.comp_loss(self.X, self.alpha)

        self.nit += 1

    def denoise(self, niter):
        for _ in range(niter):
            self.ForwardBackwardIter()
        return self.X

    def diagnostics(self, verbose=True):
        MSE_Y = np.mean((self.GT-self.Y)**2)
        MSE_X = np.mean((self.GT-self.X)**2)
        if verbose:
            print('MSE(Y) = {}'.format(MSE_Y))
            print('MSE(X) = {}'.format(MSE_X))

        # Compute unweighted ellipticities
        ell_Y = FindEll(self.Y,self.U)
        ell_X = FindEll(self.X,self.U)
        ell_GT = FindEll(self.GT,self.U)

        if verbose:
            print("Unweighted ellipticity differences :")
            print("Difference between Original and Y : {}".format(norm2(ell_GT-ell_Y)))
            print("Difference between Original and X : {}".format(norm2(ell_GT-ell_X)))
            print("Difference between X and Y : {}\n".format(norm2(ell_X-ell_Y)))
        return [MSE_Y, MSE_X], [ell_Y, ell_X, ell_GT]
