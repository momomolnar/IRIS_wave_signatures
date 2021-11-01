#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:26:51 2020

Some routines to be used with IBIS data;

@author: molnarad
"""

import numpy as np
import scipy as sp
import matplotlib as mp
import matplotlib.pyplot as pl
from scipy import io
from scipy import signal

#TURBO CHARGE YOUR PYTHON -- Make it parallel 
import multiprocessing 
#from joblib import Parallel, delayed 

import matplotlib.colors as mcolors

from numpy.random import multivariate_normal

#from skimage.transform import downscale_local_mean



def count_zeroes_in_array(array):
    '''
    Count number of zeroes in a 0-1 mask.
    
    Parameters
    ----------
    array : np.ndarray
        Input Array mask.

    Returns
    -------
    counter : int
        Number of zeroes in the array.

    '''
    counter = 0
    for el in array.ravel():
        if el == False: 
            counter += 1
        
    return counter

def load_ALMA_Band3(data_dir, time_block):
    '''
    Parameters
    ----------
    time_block : int
        Which timeblock of the ALMA dataset to be used for the analysis (2-5);
    Returns
    -------
    freqs: float array
        Frequency grid 
    ALMA6_fft_cut: float array
        PSD of the pixels chosenc
    '''
    
    file_name  = ('ALMA_band3_t' + str(time_block) 
                  +'_csclean_feathered.sav')
    sav_file   = io.readsav(data_dir + 'ALMA/' + file_name)
    ALMA3_data = np.zeros((360, 360, 299))
    
    for ii in range(299):
        ALMA3_data[:, :, ii] = np.rot90(sav_file['maps'][ii][0].transpose())
    
    cut = 100
    dx1 = ALMA3_data.shape[0]//2 - cut
    dx2 = ALMA3_data.shape[0]//2 + cut
    ddx = dx2 - dx1

    ALMA3_fft = [signal.periodogram(el-np.mean(el))[1]
                           for line in ALMA3_data[dx1:dx2, dx1:dx2, :]
                                                   for el in line] 
                          
    ALMA3_fft_cut = np.reshape(ALMA3_fft, (ddx**2, 
                                           ALMA3_data.shape[2]//2 + 1))
    freqs    = signal.periodogram(ALMA3_data[0, 0, :], fs = 1/2.0)[0]
    freqs    = freqs * 1e3
    
    return np.array(freqs), np.array(ALMA3_fft_cut)

def load_ALMA_Band6(data_dir, time_block):
    '''
    

    Parameters
    ----------

    time_block : int
        Which timeblock of the ALMA dataset to be used for the analysis (2-5);

    Returns
    -------
    freqs: float array
        Frequency grid 
    ALMA6_fft_cut: float array
        PSD of the pixels chosenc
        

    '''
    
    file_name  = ('aligned_ALMA_20170423_band6_dc_t' + str(time_block) 
                  +'_feathered_snr3.sav')
    sav_file   = io.readsav(data_dir + 'ALMA/' + file_name)
    map_ex     = sav_file['maps'][0][0].transpose()
    ALMA6_data = np.zeros((map_ex.shape[0],
                           map_ex.shape[1],
                           238))
    
    for ii in range(238):
        ALMA6_data[:, :, ii] = np.rot90(
            sav_file['maps'][ii][0].transpose())
   
    cut = 90
    
    dx1 = ALMA6_data.shape[0]//2 - cut
    dx2 = ALMA6_data.shape[0]//2 + cut
    
    ddx = dx2 - dx1
    
    ALMA6_fft = [signal.periodogram(el-np.mean(el))[1]
                           for line in ALMA6_data[dx1:dx2, dx1:dx2, :]
                                                   for el in line] 
                          
    ALMA6_fft_cut = np.reshape(ALMA6_fft, (ddx**2, 
                                           ALMA6_data.shape[2]//2 + 1))
    
    freqs    = signal.periodogram(ALMA6_data[0, 0, :], fs = 1/2.0)[0]
    freqs    = freqs * 1e3
    
    return np.array(freqs), np.array(ALMA6_fft_cut)

def construct_name(spec_property, time):
    '''
    Generate names for the save files to be retrieved from the 
    .sav library.

    Parameters
    ----------
    spec_property :  string
        Spectral property to be investigated.
        
    time : string
        Time of observations

    Returns
    -------
    name : string
        DESCRIPTION.

    '''
    
    filter_dic = {141344:"8542", 141807:"6563", 151321:"8542", 151857:"6563", 
              153954:"8542", 154624:"6563", 170444:"8542", 171115:"6563", 
              181227:"6563", 2:"", 4:""}
    
    if int(time) < 150000:
        target = '.23Apr2017.target1.all.'
    else:
        target = '.23Apr2017.target2.all.'

    name = ("fft.nb."+ spec_property + target
            + filter_dic[time] + '.ser_'
            + str(time) + ".sav")
    return name


def load_fft_data(data_dir, spec_obs, time):
    '''
    Load the data from the .sav files
    Parameters
    ----------
    data_dir : string
        Location of the data
    spec_obs : string
        one of spec_obs
     time: int
        time of the observation to be plot

    Returns
    -------
    fft_freq: float array
        frequencies of the PSDs
    fft_data: 2D float array 
        PSDs 
    '''
    data_file     = construct_name(spec_obs, time)
    #pix_lim       = 0
    #dx            = 1
    filter_dic = {141344:"8542", 141807:"6563", 151321:"8542", 151857:"6563", 
              153954:"8542", 154624:"6563", 170444:"8542", 171115:"6563", 
              181227:"6563", 2:"", 4:""}
    
    ca_v_coeff = 1233.5
    ha_v_coeff = 2089

    
    if filter_dic[time] == "8542":
        corr_coeff = ca_v_coeff
    elif filter_dic[time] == "6563":
        corr_coeff = ha_v_coeff
    
    fft_dict = sp.io.readsav(data_dir + 'IBIS/'
                             + data_file, python_dict=True)
    fft_data = fft_dict['fft_data'] * corr_coeff
    # fft_data = downscale_local_mean(fft_dict['fft_data'], (1, 1, 1))
    #fft_data = fft_data[:, pix_lim:-pix_lim:dx, pix_lim:-pix_lim:dx]
    fft_freq = fft_dict['freq1'] * 1e3 #make it in mHz
    
    fft_data = fft_data.reshape(fft_data.shape[0],
                                fft_data.shape[1]*fft_data.shape[2])
    fft_data = np.swapaxes(fft_data, 0, 1)
    print('Loaded ' + spec_obs + f' {filter_dic[time]}')
    
    return np.array(fft_freq), np.array(fft_data)

def load_fft_data_simple(data_dir, spec_obs, time):
    '''
    Load the data from the .sav files into a 1000x1000xfreq_n cube
    Parameters
    ----------
    data_dir : string
        Location of the data
    spec_obs : string
        one of spec_obs
     time: int
        time of the observation to be plot

    Returns
    -------
    fft_freq: float array
        frequencies of the PSDs
    fft_data: 2D float array 
        PSDs 
    '''
    data_file     = construct_name(spec_obs, time)

    filter_dic = {141344:"8542", 141807:"6563", 151321:"8542", 151857:"6563", 
              153954:"8542", 154624:"6563", 170444:"8542", 171115:"6563", 
              181227:"6563", 2:"", 4:""}
    
    ca_v_coeff = 1233.5
    ha_v_coeff = 2089

    
    if filter_dic[time] == "8542":
        corr_coeff = ca_v_coeff
    elif filter_dic[time] == "6563":
        corr_coeff = ha_v_coeff
    
    
    fft_dict = sp.io.readsav(data_dir + 'IBIS/'
                             + data_file, python_dict=True)
    fft_data = fft_dict['fft_data'] * corr_coeff
    fft_freq = fft_dict['freq1'] * 1e3 #make the frequencies in mHz
    dfreq    = fft_freq[1]
    fft_data = fft_data / dfreq
    print('Loaded ' + spec_obs + f' {filter_dic[time]}')
    
    return np.array(fft_freq), np.array(fft_data)
    