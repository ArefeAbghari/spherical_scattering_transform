#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:38:42 2021

@author: arefe
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import healpy as hp
import pixell as px
from pixell import reproject
from classy import Class


cmap = plt.cm.RdBu

try:
    cosmo.struct_cleanup()
except:
    pass
cosmo = Class()
cosmo.set({
    'output': 'tCl lCl', 
    'modes': 's',
    'omega_b': 0.022383, 
    'omega_cdm': 0.12011, 
    '100*theta_s': 1.041085, 
    'tau_reio': 0.0543, 
    'A_s': np.exp(3.0448)*1e-10, 
    'n_s': 0.96605, 
    'm_ncdm': 0.06, 
    'N_ncdm': 1, 
    'N_ur': 2.0328, 
    'lensing': 'yes', 
    'l_max_scalars': 20000
})

cosmo.compute()


ell = cosmo.lensed_cl()['ell']
lmax = ell[-1]
norm = ell * (ell+1) / (2*np.pi)
cl_cmb = cosmo.lensed_cl()['tt'] * cosmo.T_cmb()**2 / 1e-6**2


cl_pwl = ell**2 / (1e9 + ell**4) * 2e8
cl_pwl[1:] /= norm[1:]

#plt.style.use(['mysize_meeting'])
fig = plt.figure()
fw = fig.get_figwidth()
fh = fig.get_figheight()


fig = plt.figure(figsize=(fw, 1.5*fh))
ax1 = plt.subplot(211)
ax1.plot(ell, cl_cmb*norm, c='k',   label="CMB power spectrum (from CLASS with P18 best-fit parameters)")
ax1.plot(ell, cl_pwl*norm, c='0.5', label="custom power-laws power spectrum")
ax1.legend(handlelength=0, labelcolor='linecolor')

ax2 = plt.subplot(212)
ax2.loglog(ell, cl_cmb*norm, c='k'  )
ax2.loglog(ell, cl_pwl*norm, c='0.5')
ax2.set_ylim((cl_pwl*norm)[2]/10, (cl_pwl*norm).max()*4)


# =============================================================================
# 
# alm_cmb = hp.synalm(cl_cmb, lmax=lmax)
# alm_pwl = hp.synalm(cl_pwl, lmax=lmax)
# 
# map_cmb_1024 = hp.alm2map(alm_cmb, nside=1024, lmax=lmax)
# 
# 
# =============================================================================
map_cmb_1024 = hp.synfast(cl_cmb , nside=1024, lmax=lmax)

vmax = max(abs(map_cmb_1024.min()), abs(map_cmb_1024.max()))
hp.mollview(map_cmb_1024, min=-vmax, max=vmax, cmap=cmap, title="CMB map")




ell, DlTT = np.loadtxt("CAMB_fiducial_cosmo_scalCls.dat", usecols=(0, 1), unpack=True)
plt.plot(ell,DlTT)
plt.ylabel('$D_{\ell}$ [$\mu$K$^2$]')
plt.xlabel('$\ell$')
plt.show()


#DlTT = cl_cmb*norm
## variables to set up the size of the map
N = 2**10  # this is the number of pixels in a linear dimension
            ## since we are using lots of FFTs this should be a factor of 2^N
pix_size  = 0.5 # size of a pixel in arcminutes

## variables to set up the map plots
c_min = -400  # minimum for color bar
c_max = 400   # maximum for color bar
X_width = N*pix_size/60.  # horizontal map width in degrees
Y_width = N*pix_size/60.  # vertical map width in degrees
def make_CMB_T_map(N,pix_size,ell,ClTT):
    "makes a realization of a simulated CMB sky map given an input DlTT as a function of ell," 
    "the pixel size (pix_size) required and the number N of pixels in the linear dimension."
    #np.random.seed(100)
    # convert Dl to Cl
    #ClTT = DlTT * 2 * np.pi / (ell*(ell+1.))
    ClTT[0] = 0. # set the monopole and the dipole of the Cl spectrum to zero
    ClTT[1] = 0.

    # make a 2D real space coordinate system
    onesvec = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.) # create an array of size N between -0.5 and +0.5
    # compute the outer product matrix: X[i, j] = onesvec[i] * inds[j] for i,j 
    # in range(N), which is just N rows copies of inds - for the x dimension
    X = np.outer(onesvec,inds) 
    # compute the transpose for the y dimension
    Y = np.transpose(X)
    # radial component R
    R = np.sqrt(X**2. + Y**2.)
    # now make a 2D CMB power spectrum
    pix_to_rad = (pix_size/60. * np.pi/180.) # going from pix_size in arcmins to degrees and then degrees to radians
    ell_scale_factor = 2. * np.pi /pix_to_rad  # now relating the angular size in radians to multipoles
    ell2d = R * ell_scale_factor # making a fourier space analogue to the real space R vector
    ClTT_expanded = np.zeros(int(ell2d.max())+1) 
    # making an expanded Cl spectrum (of zeros) that goes all the way to the size of the 2D ell vector
    ClTT_expanded[0:(ClTT.size)] = ClTT # fill in the Cls until the max of the ClTT vector

    # the 2D Cl spectrum is defined on the multiple vector set by the pixel scale
    CLTT2d = ClTT_expanded[ell2d.astype(int)] 
    #plt.imshow(np.log(CLTT2d))
        
    

    # now make a realization of the CMB with the given power spectrum in real space
    random_array_for_T = np.random.normal(0,1,(N,N))
    FT_random_array_for_T = np.fft.fft2(random_array_for_T)   # take FFT since we are in Fourier space 
    
    FT_2d = np.sqrt(CLTT2d) * FT_random_array_for_T # we take the sqrt since the power spectrum is T^2
    plt.imshow(np.real(FT_2d))
        
    
    ## make a plot of the 2D cmb simulated map in Fourier space, note the x and y axis labels need to be fixed
    #Plot_CMB_Map(np.real(np.conj(FT_2d)*FT_2d*ell2d * (ell2d+1)/2/np.pi),0,np.max(np.conj(FT_2d)*FT_2d*ell2d * (ell2d+1)/2/np.pi),ell2d.max(),ell2d.max())  ###
    
    # move back from ell space to real space
    CMB_T = np.fft.ifft2(np.fft.fftshift(FT_2d)) 
    # move back to pixel space for the map
    CMB_T = CMB_T/(pix_size /60.* np.pi/180.)
    # we only want to plot the real component
    CMB_T = np.real(CMB_T)
    ## return the map
    return(CMB_T)
  ###############################

def Plot_CMB_Map(Map_to_Plot,c_min,c_max,X_width,Y_width):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    print("map mean:",np.mean(Map_to_Plot),"map rms:",np.std(Map_to_Plot))
    plt.gcf().set_size_inches(10, 10)
    im = plt.imshow(Map_to_Plot, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r)
    im.set_clim(c_min,c_max)
    ax=plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im, cax=cax)
    #cbar = plt.colorbar()
    im.set_extent([0,X_width,0,Y_width])
    plt.ylabel('angle $[^\circ]$')
    plt.xlabel('angle $[^\circ]$')
    cbar.set_label('tempearture [uK]', rotation=270)
    
    plt.show()
    return(0)
  ###############################

## make a CMB T map
CMB_T = make_CMB_T_map(N,pix_size,ell,cl_cmb)
Plot_CMB_Map(CMB_T,c_min,c_max,X_width,Y_width)
plt.savefig('cmb1.png')
plt.clf()










import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from kymatio import Scattering2D
from PIL import Image
import os


