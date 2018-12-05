        #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:05:30 2018

@author: fnammour
"""

#%%DATA INITIALIZATION
import numpy as np
import matplotlib.pyplot as plt
from modopt.signal.wavelet import get_mr_filters, filter_convolve
from AlphaTransform import AlphaShearletTransform as AST
from genU import makeUi

def hard_thresh(signal, threshold):
    return signal*(np.abs(signal)>=threshold)

def sigma_mad(signal):
    return 1.4826*np.median(np.abs(signal-np.median(signal)))

def MS_hard_thresh(wave_coef, n_sigma):
    wave_coef_rec_MS = np.zeros(wave_coef.shape)
    for i,wave in enumerate(wave_coef):
        # Denoise image
        wave_coef_rec_MS[i,:,:] = hard_thresh(wave, n_sigma[i])
    return wave_coef_rec_MS

def norm2(signal):
    return np.linalg.norm(signal,2)

def norm1(signal):
    return np.linalg.norm(signal,1)

def norm(signal):
    return np.linalg.norm(signal)

def get_adjoint_coeff():
    column = trafo.width
    row = trafo.height
    n_scales = len(trafo.indices)
    #Attention: the type of the output of trafo.adjoint_transform is complex128
    #and by creating coeff without specifying the type it is set to float64
    #by default when using np.zeros
    coeff = np.zeros((n_scales,row,column))
    for s in range(n_scales):
        temp = np.zeros((n_scales,row,column))
        temp[s,row//2,column//2]=1
        coeff[s] = trafo.adjoint_transform(temp, do_norm=False)
    return coeff

def shear_norm(signal,shearlets):
    shearlet_norms = np.array([norm(s) for s in shearlets])
    return np.array([s/n for s,n in zip(signal,shearlet_norms)])

def scal(a,b):
    return (a*b).sum()

def G(X,U,W = 1):
    return np.array([scal(X*W,U[i]) for i in range(6)])

def prior(alpha):
    norm = 0
    for wave in alpha:
        norm += norm1(wave)
    return norm 

def comp_adj(imgs,adjoints):
    return np.array([filter_convolve(i,adjoints) for i in imgs])

def comp_mu(adj):
    mu = np.array([[1/norm(im)**2 if norm(im)!=0 else 0 for im in u]
                                                            for u in adj])
    return mu/(6*mu[mu!=0].size)

def comp_grad(X,Y,adj_U,mu):
    return gamma*np.array([[2*cst*scal(X-Y,im)*im 
                            for cst,im in zip(m, u)]
                             for m,u in zip(mu,adj_U)]).sum((0,1)) + X - Y

def comp_thresh(alpha,k=4):
    thresholds = []
    thresholds += [(k+1)*sigma_mad(alpha[0])]
    for wave in alpha[1:-1]:
        thresholds += [k*sigma_mad(wave)]
    return np.array(thresholds)

def update_thresholds(X,Y,filters,thresholds,k,itr,first_run):
    R = Y - X
    alphaR = filter_convolve(R, filters)
    if first_run and itr < 5:
        thresholds = comp_thresh(alphaR,k)    
    return thresholds

def comp_loss(X,Y,alpha,gamma,mu,adj_U):
    return np.array([norm(X - Y)**2/2.,gamma*(np.array(
            [[cst*(((X-Y)*im).sum())**2*im for cst,im in zip(m, u)]
            for m,u in zip(mu,adj_U)])/2.).sum(),prior(alpha)])

def reconstruct(alpha, positivity = True):
    X = alpha.sum(0)
    if positivity:
        X = X*(X>0)
    return X

def FindEll(X, U, W = 1):
    GX = G(X,U,W)
    mu20 = 0.5*(GX[3]+GX[4])-GX[0]**2/GX[2]
    mu02 = 0.5*(GX[3]-GX[4])-GX[1]**2/GX[2]
    mu11 = GX[5]-GX[0]*GX[1]/GX[2]
    e1 = (mu20-mu02)/(mu20+mu02)
    e2 = 2*(mu11)/(mu20+mu02)
    return np.array([e1,e2])

#Load data
SNR = 10

gals = np.load('Denoising_SNR{}/convolved_galaxiesNO.npy'.format(SNR))
noisy_gals = np.load('Denoising_SNR{0}/noisy_galaxies_SNR{0}.npy'.format(SNR))

#DENOISING

#Initialisation
gal_num = 12

GT = gals[gal_num]#convolved ground truth galaxy
Y = noisy_gals[gal_num]#observed galaxy
X = np.ones(Y.shape)/Y.size #initial estimate

_,row,column = noisy_gals.shape
U = makeUi(row,column)

gamma = 9 #trade-off parameter

#Get filters of transforms
filters = get_mr_filters((row,column), coarse = True)# Get starlet filters
trafo = AST(Y.shape[1], Y.shape[0], [0.5]*3,real=True,parseval=True,verbose=False)# Get shearlet filters
shearlets = trafo.shearlets
adjoints = get_adjoint_coeff()
#Normalize shearlets filter banks
adjoints = shear_norm(adjoints,shearlets)
shearlets = shear_norm(shearlets,shearlets)

alpha = filter_convolve(X, filters) #Starlet transform of X
alphaY = filter_convolve(Y, filters) #Starlet transform of Y

omega = trafo.transform(X) #Shearlet transform of X
omegaY = trafo.transform(Y) #Shearlet transform of Y

#Compute moments constraint normalization coefficients
adj_U = comp_adj(U,adjoints)
mu = comp_mu(adj_U)

#Lipschitz constant of the gradient
L = 1 + 2*gamma#*np.array([mu[i]*norm(adj_U[i])**2 for i in range(6)]).sum()

k = 5 #Set k for k sigma_mad

#Calculate thresholds using Y, the observed galaxy
thresholds = comp_thresh(alphaY,k)

beta = 1/L

first_run = True

#%%LOOPING

niter = 150 #number of iterations
loss = np.zeros((2*niter+1,3))

if first_run:
    niter_tot = niter
    loss_tot = loss
    loss[0] = comp_loss(X,Y,alpha,gamma,mu,adj_U)
else:
    loss[0] = loss_tot[2*niter_tot]

notconv=True
epsilon = 0.01

itr=0

while itr<niter and notconv:
    #One iteration

    #Fidelity gradient according to alpha
    grad = comp_grad(X,Y,adj_U,mu)
    
    #Update X
    X = X- beta*grad
    
    #Compute LOSS for the first half of the iteration
    loss[2*itr+1] = comp_loss(X,Y,alpha,gamma,mu,adj_U)

    #Starlet transform of X
    alpha = filter_convolve(X, filters)

    #Update thresholds
    thresholds = update_thresholds(X,Y,filters,thresholds,k,itr,first_run)
    
    #Multiscale threshold except coarse scale
    alpha[:-1] = MS_hard_thresh(alpha[:-1], beta*np.array(thresholds))
    
    #Reconstrcut X
    X = reconstruct(alpha)
    
    #Calculate loss for the second half of the iteration
    loss[2*(itr+1)] = comp_loss(X,Y,alpha,gamma,mu,adj_U)
    
    if norm(loss[2*itr]-loss[2*(itr+1)])<epsilon:
        notconv=False
    
    itr += 1

if first_run:
    loss_tot = loss
    first_run = False
else:
    loss_tot = np.vstack((loss_tot,loss[1:]))
    niter_tot += niter

#Print galaxy
colormap = 'viridis'
interpol = 'Nearest'    

print('Denoising SNR = {} ({} iterations)'.format(SNR,niter))

plt.figure(1)
plt.imshow(GT,cmap = colormap,interpolation =interpol)
plt.colorbar()
plt.title(r'$X$ Ground truth')

plt.figure(2)
plt.imshow(Y,cmap = colormap,interpolation =interpol)
plt.colorbar()
plt.title(r'$Y$ Observed')

plt.figure(3)
plt.imshow(X,cmap = colormap,interpolation =interpol)
plt.colorbar()
plt.title(r'$\hat{X}$ Denoised')

plt.figure(4)
plt.imshow(np.abs(GT-X),cmap = colormap,interpolation =interpol)
plt.colorbar()
plt.title(r'$|X-\hat{X}|$')

plt.figure(5)
plt.imshow(np.abs(Y-X),cmap = colormap,interpolation =interpol)
plt.colorbar()
plt.title(r'$|Y-\hat{X}|$')

MSE_Y = np.mean((GT-Y)**2)
MSE_X = np.mean((GT-X)**2)

print('MSE(Y) = {}'.format(MSE_Y))
print('MSE(X) = {}'.format(MSE_X))

#Plot the mean LOSS funcitons
absciss = np.arange(2*niter_tot+1)/2
plt.figure(6)
plt.title('LOSS functions SNR = {} ({} iterations)'.format(SNR,niter))
plt.semilogy(absciss,loss_tot[:,0],'r', label = 'Fidelity')
plt.semilogy(absciss,loss_tot[:,1],'b', label = 'Moments contraint')
plt.semilogy(absciss,loss_tot[:,2],'y', label = 'Sparsity constraint')
plt.semilogy(absciss,np.sum(loss_tot,1), 'g', label = 'Total LOSS')
plt.legend()
plt.show()

plt.figure(7)
plt.title('Mean differential LOSS functions SNR = {} ({} iterations)'.format(SNR,niter))
plt.plot(absciss,loss_tot[:,0],'r', label = 'Fidelity')
plt.plot(absciss,loss_tot[:,1],'b', label = 'Moments contraint')
plt.plot(absciss,loss_tot[:,2],'y', label = 'Sparsity constraint')
plt.plot(absciss,np.sum(loss_tot,1), 'g', label = 'Total LOSS')
plt.legend()
plt.show()

#EVALUATING RESULTS

# Compute unweighted ellipticities
ell_Y = FindEll(Y,U)
ell_X = FindEll(X,U)
ell_GT = FindEll(GT,U)

print("Unweighted ellipticity differences :")
print("Difference between Original and Y : {}".format(norm2(ell_GT-ell_Y)))
print("Difference between Original and X : {}".format(norm2(ell_GT-ell_X)))
print("Difference between X and Y : {}\n".format(norm2(ell_X-ell_Y)))

#Plot unweighted ellipticities
plt.figure(8)
plt.suptitle('Unweighted $e_1$ comparaison SNR = {} ({} iterations)'.format(SNR,niter))
plt.subplot(311)
plt.scatter(ell_GT[0],ell_Y[0])
plt.plot([-1,1],[-1,1],'r')
plt.xlabel('True')
plt.ylabel('Y')
plt.title('True vs Y')
plt.subplot(312)
plt.scatter(ell_GT[0],ell_X[0])
plt.plot([-1,1],[-1,1],'r')
plt.xlabel('True')
plt.ylabel('X')
plt.title('True vs X')
plt.subplot(313)
plt.scatter(ell_Y[0],ell_X[0])
plt.plot([-1,1],[-1,1],'r')
plt.xlabel('Y')
plt.ylabel('X')
plt.title('X vs Y')
plt.subplots_adjust(wspace=1.2, hspace=1.4)
plt.show()