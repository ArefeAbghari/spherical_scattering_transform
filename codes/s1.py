#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:54:39 2021

@author: arefe
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import pandas as pd
import healpy as hp
import time
import warnings
#from pixell import reproject
#from classy import Class

def gabor(freq,sigma,theta):
    
    arg=-(theta**2)/(2*sigma*sigma) + 1.j * freq*theta
    
    g=np.exp(arg)
    g/=2*np.pi*sigma*sigma
    return g


def morlet (f, sigma, theta, lmax):
    wv=gabor(f,sigma,theta)
    wvm=gabor(0,sigma, theta)
    B=hp.beam2bl(wv,theta,lmax)[0]/hp.beam2bl(wvm,theta,lmax)[0]
    mor=wv-B*wvm
    return mor


def morlet_arr (resol, jmax, lmax, theta_bin) :
    morl=[]
    fl2beam=[]
    
    theta=np.linspace(0,np.pi,theta_bin)
    for j in range(jmax):
        sigma1 = (0.8*resol*2**j)
        freq1=(3.0*np.pi) /(4.0*resol*2**j)
        morf = morlet (freq1, sigma1, theta, lmax)
        mor_l = hp.beam2bl(morf.real, theta, lmax)
        
        morl.append(mor_l)
    
    return morl


def gaus_arr (resol, jmax, lmax, theta_bin):
    
    theta=np.linspace(0,np.pi,theta_bin)    
    
    phifilter=[]
    
    for j in range(jmax):
        sigma1 = (0.8*resol*2**j)
        freq1=(3.0*np.pi) /(4.0*resol*2**j)
        
        phi1=gabor(0,sigma1, theta)
        gab_l=hp.gauss_beam(2*np.sqrt(2*np.log(2))*sigma1,lmax)
        #gab_lb=hp.beam2bl(phi1.real,theta,lmax=lmax)
        
        phifilter.append(gab_l)
        
    return phifilter


def compS1 (hmap, mor_l, gaus_l, resol, jmax, lmax, nside ):

    S1=np.zeros((jmax))
    i1=[]
    
    mapalm=hp.map2alm(hmap,lmax,use_pixel_weights=True)
    
    #mor_l = morlet_arr(resol, jmax, lmax, jbin)
    #gaus_l = gaus_arr(resol, jmax, jbin)
    
    for j in range(jmax):
        print(j)
        sigma = (0.8*resol*2**j) # multiplied by resolution in radians
        freq=(3.0*np.pi) /(4.0*resol*2**j) #divided by resolution in radians

        #morlet filter 
        
        morl = mor_l[j]
        
        #convolving the map with filter 1

        nalm=hp.almxfl(mapalm,morl)
        
        I1=hp.alm2map(nalm,nside,lmax)
        

        #modulus
        I1=np.abs(I1)

        i1.append(I1)
        #Gaussian filter for S1
        
        gausl = gaus_l[j]
        #Convolving with gaussian filter to get S1
        I1alm=hp.map2alm(I1, lmax, use_pixel_weights=True)
        
        nI1alm=hp.almxfl(I1alm, gausl)
    
        S1map=hp.alm2map(nI1alm, nside, lmax)
    
    
        #Averaging the entire map
        S1[j]=np.mean(S1map)
        
    return S1 , i1

def compS2 ( hmap , i1 , mor_l, gaus_l  , resol, jmax, lmax, nside ):

    S2=np.zeros((jmax,jmax))

    for j1 in range(jmax):
        print (j1)
        I1=i1[j1]
        mapalm1=hp.map2alm(I1, lmax ,use_pixel_weights=True)
        
        for j2 in range (jmax):
                    
            sigma2 = (0.8*resol*2**j2)
            freq2=(3.0*np.pi) /(4.0*resol*2**j2)
            
            #filter 2        
            f2=mor_l[j2]
            
            #Convolving I1 with filter 2
            
            nalm2=hp.almxfl(mapalm1,f2)
            I2=hp.alm2map(nalm2,nside,lmax)
            #i2[j1,j2] = I2
            
            #modulus
            I2=np.abs(I2)
            
            
            #Gaussian filter to get S2
            gausl2=gaus_l[j2]
        
            #Convolving with gaussian filter to get S2
            I2alm=hp.map2alm(I2,lmax ,use_pixel_weights=True)
            nI2alm=hp.almxfl(I2alm,gausl2)
            S2map=hp.alm2map(nI2alm, nside,lmax)     
                           
            #averaging the entire map
            S2[j1][j2]=np.mean(S2map)
    
    return S2

def compS3 ( i2  , mor_l, gaus_l  , resol, jmax, lmax, nside ):

    S3 = np.zeros((jmax,jmax,jmax ))
    i3 = np.zeros ((jmax,jmax,jmax ))
    
    for j1 in range(jmax):
        
        for j2 in range (j1+1 , jmax):
            
            print (j1 , j2)
            I2=i2[j1 ][ j2 ]
            mapalm2=hp.map2alm(I2, lmax ,use_pixel_weights=True)
            
            for j3 in range (jmax):
                    
                sigma3 = (0.8*resol*2**j3)
                freq3 =(3.0*np.pi) /(4.0*resol*2**j3)
            
                #filter 3       
                f3=mor_l[j3]
            
                #Convolving I2 with filter 3
                
                nalm3 = hp.almxfl (mapalm2 , f3)
                I3 = hp.alm2map(nalm3 , nside , lmax)
                
                #modulus
                I3=np.abs(I3)
                i3 [j1 , j2 , j3] = I3
                
                if gaus_l == None:
            
                    S3map = I3
            
                else:

                    #Gaussian filter to get S2
                    gausl3 = gaus_l[j3]
                
                    #Convolving with gaussian filter to get S3
                    I3alm = hp.map2alm (I3 , lmax , use_pixel_weights=True)
                    nI3alm = hp.almxfl(I3alm , gausl3 )
                    S3map = hp.alm2map(nI3alm, nside,lmax)     
                           
            #averaging the entire map
            S3[j1][j2][j3] = np.mean(S3map)
    
    return S3 , i3