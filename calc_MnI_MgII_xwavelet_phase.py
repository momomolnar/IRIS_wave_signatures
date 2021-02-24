#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:49:49 2021
Compute the coherency of the Mn I and the Mg II h/k lines 

@author: molnarad
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from IRIS_lib import plot_velocity_map, filter_velocity_field, remove_nans
from IRIS_lib import calc_phase_diff


import pycwt as wavelet
from pycwt.helpers import find

def convert_cart_to_polar(x, y):
    """
    Convert Cartesian [x,y] to polar coordinates [r, theta]

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.

    """
    
    r     = np.sqrt(x*x + y*y)
    if y > 0:
        theta = np.arccos(x/r)
    else:
        theta = -1 * np.arccos(x/r)
    return r, theta

def convert_polar_to_cart(coords):
    x = coords[0] * np.cos(coords[1])
    y = coords[0] * np.sin(coords[1])
    return [x, y]

def angle_mean(angles):
    """
    Calculate the mean of a distribution of angles
    using the unit circle averaging.
    
    Parameters
    ----------
    angles : ndarray [num_angles]
        Input Angles array.

    Returns
    -------
    mean_angle : float
        The mean angle.

    """
    
    num_angles = len(angles)
    r = np.ones(num_angles)
    
    xy_array = [convert_polar_to_cart(el) for el in 
                zip(r, angles)]
    
    xy_array = np.array(xy_array)
    x_mean = np.mean(xy_array[:, 0])
    y_mean = np.mean(xy_array[:, 1])
    
    r, mean_angle = convert_cart_to_polar(x_mean, y_mean)
    return mean_angle

def calc_angle(timeSeries1, timeSeries2):

    angles = np.zeros((72, 123, timeSeries1.shape[0]))
    cor_sig_all = np.zeros((72, 123, timeSeries1.shape[0]))

    use_cache = True
    for el in range(40, timeSeries1.shape[0]-100):
        N = timeSeries1.shape[1]
        t = np.arange(0, N) * dt + t0

        timeSeries1_ex = timeSeries1[el, :]
        std_timeSeries1 = timeSeries1_ex.std()  # Standard deviation
        var_timeSeries1 = std_timeSeries1 ** 2  # Variance
        timeSeries1_norm = timeSeries1_ex / std_timeSeries1  # Normalized dataset

        timeSeries2_ex = timeSeries2[el, :]
        std_timeSeries2 = timeSeries2_ex.std()  # Standard deviation
        var_timeSeries2 = std_timeSeries2 ** 2  # Variance
        timeSeries2_norm =timeSeries2_ex / std_timeSeries2  # Normalized dataset

        t1 = t2 = np.linspace(0, timeSeries1_ex.size - 1, 
                              num=timeSeries1_ex.size)*dt

        mother = wavelet.Morlet(6)
        s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 secondss = 6 months
        dj = 1 / 12  # Twelve sub-octaves per octaves
        J = 7 / dj  # Seven powers of two with dj sub-octaves
        print(timeSeries1_ex.shape)
        alpha, _, _ = wavelet.ar1(timeSeries1_norm)  # Lag-1 autocorrelation for red noise


        WCT, aWCT, cor_coi, freq, sig = wavelet.wct(timeSeries1_norm, 
                                                     timeSeries2_norm, 
                                                     dt, dj=1/12, 
                                                     s0=s0, J=-1,
                                                     significance_level=0.8646,
                                                     wavelet='morlet', 
                                                     normalize=True,
                                                     cache=use_cache)

        cor_sig = np.ones([1, timeSeries1_ex.size]) * sig[:, None]
        cor_sig = np.abs(WCT) / cor_sig  # Power is significant where ratio > 1
        cor_sig_all[:, :, el] = cor_sig
        cor_period = 1 / freq
        use_cache = True
    # Calculates the phase between both time series. The phase arrows in the
    # cross wavelet power spectrum rotate clockwise with 'north' origin.
    # The relative phase relationship convention is the same as adopted
    # by Torrence and Webster (1999), where in phase signals point
    # upwards (N), anti-phase signals point downwards (S). If X leads Y,
    # arrows point to the right (E) and if X lags Y, arrow points to the
    # left (W).
        angles[:, :, el] = aWCT

    return angles, cor_sig_all, freq, cor_coi, cor_period

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/QS_data/"
filename_MnI = "iris_l2_20131116_073345_3803010103_raster_t000_r00000MnI_2801.9.sav"
filename_Mg = "iris_l2_20131116_073345_3803010103_raster_t000_r00000_mg2_vel_mu=1.0.sav"

title = 'IRIS Mg II -- Mn I'
label = 'Vel'
units = 'km s$^{-1}$'
t0 = 0
dt = 16.7 # In seconds

aa = io.readsav(d+filename_MnI)

fit_params = aa["fit_params"]

velocities = fit_params[1, :, :]

rest_wave = np.median(velocities)

velocities = (velocities - rest_wave)*3e5/2802.05
print(velocities.shape)

MnI_vel = filter_velocity_field(velocities.T,  
                                dv=7, dd=5, degree=2)

plot_velocity_map(MnI_vel, vmin=-2, vmax=2, aspect=0.3, 
                  title="Mn I 2801.5 $\AA$ Doppler velocity ")

aa = io.readsav(d+filename_Mg)

fit_params = aa["bp"]

velocities = fit_params[:, :, 0, 0]

MgII_vel = filter_velocity_field(velocities,  
                                 dv=7, dd=5, degree=2)

plot_velocity_map(MgII_vel, vmin=-7, vmax=7, aspect=0.3, 
                  title="Mg II h 2v Doppler velocity ")

angles, cor_sig_all, freq, coi_mask = calc_phase_diff(MgII_vel, MnI_vel)


angles_average = [angle_mean(el.flatten()) for el in 
                  angles * coi_mask[:, :, None]]
angles_average = np.array(angles_average)

angles_masked = angles * cor_sig_all * coi_mask[:, :, None] 


num_freq = angles_masked.shape[0]
phase_freq = np.zeros(num_freq)

for el in range(num_freq):
    try:
        temp = angles_masked[el, :, :]
        temp[np.isnan(temp)] = 0
        temp = np.ma.masked_equal(temp, 0.0)
        temp1 = temp.compressed()
        phase_freq[el] = angle_mean(temp1.flatten())
    except: 
        print(f"Too many zeros on line {el}")
        
pl.figure(dpi=250)
pl.plot(freq*1e3, angles_average*180/3.1415, label="All angles")
pl.plot(freq*1e3, phase_freq*180/3.1415, 
        label="Statistically significant")
pl.xlabel("Frequency [mHz]")
pl.title("Phase Diff Mn I - Mg II vel")
pl.grid()
pl.legend()
pl.ylabel("Phase difference [deg]")