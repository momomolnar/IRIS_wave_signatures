#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 23:56:02 2021

Calculate Doppler velocities from IRIS C II lines 
using the filtering from Rathore.

@author: molnarad
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as pl 
from astropy.io import fits
from astropy import modeling
from IRIS_lib import filter_CII_data
from skimage.transform import downscale_local_mean
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import time

fitter = modeling.fitting.LevMarLSQFitter()

def calc_Doppler_vel(waves, spectrum):
    '''
    Wrapper for Gaussian Doppler velocity fitting.

    Parameters
    ----------
    waves: ndarray [num_waves]
        Wavelength grid.
    spectrum : ndarray
        Input spectrum

    Returns
    -------
    velocity : ndarray
        output velocity

    '''
    
    max_I = np.amax(spectrum)
    mean_est = np.mean(waves)
    std_est  = (waves[1] - waves[0]) * 3
    model = modeling.models.Gaussian1D(amplitude=max_I, mean=mean_est, 
                                       stddev=std_est)
    model = fitter(model, waves, spectrum)
    
    
    return model.mean[0]

def calc_Doppler_vel_column(waves, spectrum_column, wave_limits):
    """
    Wrapper of calc_Doppler_vel for a row of spectra

    Parameters
    ----------
    waves : TYPE
        DESCRIPTION.
    spectrum_column : TYPE
        DESCRIPTION.

    Returns
    -------
    velocities : TYPE
        DESCRIPTION.

    """
    
    N_spectra = spectrum_column.shape[0]
    velocities = [calc_Doppler_vel(waves, spectrum[wave_limits[0]
                                                   :wave_limits[1]]) 
                  for spectrum in spectrum_column]
    velocities = np.array(velocities)
    
    return velocities

data_dir = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Plage_data/"
hdu = fits.open(data_dir
                + "iris_l2_20140918_101908_3820259153_raster_t000_r00000.fits")

plot_example = False
wave_limits = [62, 80]
dwave = 0.0258
num_waves = wave_limits[1] - wave_limits[0]
waves = np.linspace(0, num_waves*dwave, num=num_waves)

CII_data_raw = hdu[1].data
CII_data_downscaled = downscale_local_mean(CII_data_raw, (1, 3, 1)) #rebin 3 down in 

CII_vel = np.zeros((CII_data_downscaled.shape[0],
                    CII_data_downscaled.shape[1]))
# depending on the data you need to give some initial values
start = time.time()
CII_vel = Parallel(n_jobs=4)(delayed(calc_Doppler_vel_column)(waves, spectrum,
                                                              wave_limits) 
                                       for spectrum in 
                                       CII_data_downscaled[:, :, :])
end = time.time()
print(f"Time for running {end - start:.3f}")

mean_vel = np.mean(CII_vel, axis=0)
CII_vel -= mean_vel[None, :]
CII_vel *= 3e5/1335
#for el in tqdm(range(0, CII_data_downscaled.shape[0]), position=0, leave=True):
#
#    CII_data_temp = filter_CII_data(CII_data_downscaled[el, :,
#                                                        wave_limits[0]:wave_limits[1]])
#    CII_data_temp = CII_data_downscaled[el, :, wave_limits[0]:wave_limits[1]]
#CII_vel[el, :] = Parallel(n_jobs=4)(delayed(calc_Doppler_vel)(waves, spectrum) 
#                                    for spectrum in CII_data_temp)
#    CII_vel[el, :] = np.array([calc_Doppler_vel(waves, data) for data in 
#                               CII_data_temp])





if plot_example == True: 
    CII_data_temp = CII_data_downscaled[100, :, wave_limits[0]:wave_limits[1]]

    CII_vel[el, :] = np.array([fitter(model, waves, 
                                      data).mean[0] for data in 
                               CII_data_temp])
    pl.plot(waves, CII_data_temp[100, 50,
                                 wave_limits[0]:wave_limits[1]], 
            'r.', label="data")
    pl.plot(waves, norm.pdf(waves,), 'b--', label="model")
    pl.legend()
    pl.show()
    
    