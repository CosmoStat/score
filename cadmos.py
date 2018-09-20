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
from genU import makeUi
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# Sam's custom cmap
colors = [(0, 0, 1), (1, 0, 0), (1, 1, 0)]  # B -> R -> Y
Samcmap = LinearSegmentedColormap.from_list('my_colormap', colors)
Samcmap.set_under('k')

def plot_func(im, wind=False, cmap='gist_stern', norm=None, cutoff=5e-4,filename = False):
    if cmap in ['sam','Sam']:
        cmap = Samcmap
        boundaries = np.arange(cutoff, np.max(im), 0.0001)
        norm = BoundaryNorm(boundaries, plt.cm.get_cmap(name=cmap).N)
    if len(im.shape) == 2:
        if not wind:
            plt.imshow(im, cmap=cmap, norm=norm,
                       interpolation='Nearest')
        else:
            vmin, vmax = wind
            plt.imshow(im, cmap=cmap, norm=norm,
                       interpolation='Nearest', vmin=vmin, vmax=vmax)
    else:
        sqrtN = int(np.sqrt(im.shape[0]))
        if not wind:
            plt.imshow(im.reshape(sqrtN,sqrtN), cmap=cmap, norm=norm,
                       interpolation='Nearest')
        else:
            vmin, vmax = wind
            plt.imshow(im.reshape(sqrtN,sqrtN), cmap=cmap, norm=norm, 
                       interpolation='Nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    if not(filename):
        plt.show()
    else:
        plt.savefig(filename)

def save_fig(im,title,filename):
    plt.title(title)
    plot_func(im,cmap='Sam',filename = filename)
    plt.close()

def soft_thresh(signal, threshold):
    return np.sign(signal)*(np.abs(signal)-threshold)*(np.abs(signal)>=threshold)

def hard_thresh(signal, threshold):
    return signal*(np.abs(signal)>=threshold)

def sigma_mad(signal):
    return 1.4826*np.median(np.abs(signal-np.median(signal)))

def MS_soft_thresh(wave_coef, n_sigma):
    wave_coef_rec_MS = np.zeros(wave_coef.shape)
    for i,wave in enumerate(wave_coef):
        # Denoise image
        wave_coef_rec_MS[i,:,:] = soft_thresh(wave, n_sigma[i])
    return wave_coef_rec_MS

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

def G(X,U,W = 1):
    return np.array([(X*W*U[i]).sum() for i in range(6)])

def prior(alpha):
    norm = 0
    for wave in alpha:
        norm += norm1(wave)
    return norm

def comp_mu(W,U):
    return np.array([1/(6*norm2(W*U[i])**2) for i in range(6)])

def comp_grad(X,Y,W,U,mu):
    return gamma*W*np.array([mu[i]*G(X-Y,U,W)[i]*U[i] for i in range(6)]).sum(0) + X - Y

def comp_thresh(alpha,k=4):
    thresholds = []
    thresholds += [(k+1)*sigma_mad(alpha[0])]
    for wave in alpha[1:-1]:
        thresholds += [k*sigma_mad(wave)]
    return np.array(thresholds)

def update_thresholds(X,Y,filters,thresholds,k,itr,first_run):
    R = Y - X
    alphaR = filter_convolve(R, filters)
    if first_run and itr < 5000:
        thresholds = comp_thresh(alphaR,k)    
    return thresholds

def comp_loss(X,Y,W,alpha,gamma,mu,U):
    return np.array([norm2(X - Y)**2/2.,gamma*(mu*G(X-Y,U,W)**2/2.).sum(),prior(alpha)])

def update_beta(X,Y,grad,loss,beta,sigma,epsilon,thresholds,filters,first_run,itr,gamma,mu,U,W,k):
    sigma = float(sigma)
    #Calculate candidate X^{t+1/2}
    temp = X- beta*grad

    alphaT = filter_convolve(temp, filters)
    thresholds = update_thresholds(temp,Y,filters,thresholds,k,itr,first_run)
    alphaT[:-1] = MS_hard_thresh(alphaT[:-1], beta*np.array(thresholds))
    temp = reconstruct(alphaT)
    loss1 = comp_loss(temp,Y,W,alphaT,gamma,mu,U)
    
    #Update beta
    if beta>epsilon:
        while loss1.sum() > loss.sum():
            beta = beta/sigma
            temp = X- beta*grad
            
            alphaT = filter_convolve(temp, filters)
            thresholds = update_thresholds(temp,Y,filters,thresholds,k,itr,first_run)
            alphaT[:-1] = MS_hard_thresh(alphaT[:-1], beta*np.array(thresholds))
            temp = reconstruct(alphaT,k)
            
            loss1 = comp_loss(temp,Y,W,alphaT,gamma,mu,U)
            if beta < epsilon:
                break
    return beta

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
gauss_win = np.load('Denoising_SNR{}/gaussian_windows.npy'.format(SNR))
noisy_gals = np.load('Denoising_SNR{0}/noisy_galaxies_SNR{0}.npy'.format(SNR))

#DENOISING

#Initialisation
gal_num = 54

GT = gals[gal_num]#convolved ground truth galaxy
Y = noisy_gals[gal_num]#observed galaxy
X = np.ones(Y.shape)#/Y.size #initial estimate
W = gauss_win[gal_num]

_,row,column = noisy_gals.shape
U = makeUi(row,column)

#Moment constraints normalization
mu = comp_mu(W,U)

gamma = 10.2 #trade-off parameter

#Lipschitz constant of the gradient
L = 1 + gamma*np.array([mu[i]*norm2(W*U[i])**2 for i in range(6)]).sum()

filters = get_mr_filters((row,column), coarse = True)# Get wavelet filters
alpha = filter_convolve(X, filters) #Starlet transform of X
alphaY = filter_convolve(Y, filters) #Starlet transform of Y

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
    loss[0] = comp_loss(X,Y,W,alpha,gamma,mu,U)
else:
    loss[0] = loss_tot[2*niter_tot]

for itr in range(niter):
    #One iteration

    #Fidelity gradient according to alpha
    grad = comp_grad(X,Y,W,U,mu)
    
    #Update X
    X = X- beta*grad
    
    #Compute LOSS for the first half of the iteration
    loss[2*itr+1] = comp_loss(X,Y,W,alpha,gamma,mu,U)

    #Starlet transform of X
    alpha = filter_convolve(X, filters)

    #Update thresholds
    thresholds = update_thresholds(X,Y,filters,thresholds,k,itr,first_run)
    
    #Multiscale threshold except coarse scale
    alpha[:-1] = MS_hard_thresh(alpha[:-1], beta*np.array(thresholds))
    
    #Reconstrcut X
    X = reconstruct(alpha)
    
    #Calculate loss for the second half of the iteration
    loss[2*(itr+1)] = comp_loss(X,Y,W,alpha,gamma,mu,U)

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

#load true ellipticities
ell_true = np.load('./Denoising_SNR{}/true_ellipticitiesNO.npy'.format(SNR))
ell_true = ell_true[gal_num,:]

# Compute unweighted ellipticities
ell_Y = FindEll(Y,U)
ell_X = FindEll(X,U)
ell_GT = FindEll(GT,U)

print("Unweighted ellipticity differences :")
print("Difference between Original and Y : {}".format(norm2(ell_GT-ell_Y)))
print("Difference between Original and X : {}".format(norm2(ell_GT-ell_X)))
print("Difference between X and Y : {}\n".format(norm2(ell_X-ell_Y)))

# Compute weighted ellipticities
ell_WY = FindEll(Y,U,W)
ell_WX = FindEll(X,U,W)
ell_WGT = FindEll(GT,U,W)

print("Weighted ellipticity differences :")
print("Difference between weighted Original and WY : {}".format(norm2(ell_WGT-ell_WY)))
print("Difference between weighted Original and WX : {}".format(norm2(ell_WGT-ell_WX)))
print("Difference between WX and WY : {}".format(norm2(ell_WX-ell_WY)))

#Plot unweighted ellipticities
plt.figure(8)
plt.suptitle('Unweighted $e_1$ comparaison SNR = {} ({} iterations)'.format(SNR,niter))
plt.subplot(311)
plt.scatter(ell_true[0],ell_Y[0])
plt.plot([-1,1],[-1,1],'r')
plt.xlabel('True')
plt.ylabel('Y')
plt.title('True vs Y')
plt.subplot(312)
plt.scatter(ell_true[0],ell_X[0])
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

#Plot ellipticities
plt.figure(9)
plt.suptitle('Weighted $e_1$ comparaison SNR = {} ({} iterations)'.format(SNR,niter))
plt.subplot(311)
plt.scatter(ell_true[0],ell_WY[0])
plt.plot([-1,1],[-1,1],'r')
plt.xlabel('True')
plt.ylabel('WY')
plt.title('True vs WY')
plt.subplot(312)
plt.scatter(ell_true[0],ell_WX[0])
plt.plot([-1,1],[-1,1],'r')
plt.xlabel('True')
plt.ylabel('WX')
plt.title('True vs WX')
plt.subplot(313)
plt.scatter(ell_WY[0],ell_WX[0])
plt.plot([-1,1],[-1,1],'r')
plt.xlabel('WY')
plt.ylabel('WX')
plt.title('WX vs WY')
plt.subplots_adjust(wspace=1.2, hspace=1.4)
plt.show()
