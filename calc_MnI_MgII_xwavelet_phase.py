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
from IRIS_lib_coherency import plot_velocity_map, filter_velocity_field 
from IRIS_lib_coherency import remove_nans

from IRIS_lib import calc_phase_diff, angle_mean, calc_phase_diff_full

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

data_dir = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Sample_datasets/"

Plage_data = "Plage_int_vel_20140918_101908.npz"
QS_data    = "QS_int_vel_2131116_073345.npz"

with np.load(data_dir + QS_data) as QS_data:
    qs_cad = QS_data["qs_cad"]
    intensity_QS_MnI_rel  = QS_data["intensity_QS_MnI_rel"]
    intensity_QS_MgII_rel = QS_data["intensity_QS_MgII_rel"].T
    intensity_QS_MnI_zero_mean = QS_data["intensity_QS_MnI_zero_mean"]
    intensity_QS_MgII_zero_mean = QS_data["intensity_QS_MgII_zero_mean"].T
    vel_reduced_QS_MnI    = QS_data["vel_reduced_QS_MnI"].T
    vel_reduced_QS_MgII   = QS_data["vel_reduced_QS_MgII"].T
    
with np.load(data_dir + Plage_data) as QS_data:
    plage_cad = QS_data["plage_cad"]
    intensity_P_MnI_rel  = QS_data["intensity_P_MnI_rel"]
    intensity_P_MgII_rel = QS_data["intensity_P_MgII_rel"].T
    intensity_P_MgII_zero_mean = QS_data["intensity_P_MgII_zero_mean"].T
    intensity_P_MnI_zero_mean = QS_data["intensity_P_MnI_zero_mean"]
    vel_reduced_P_MnI    = QS_data["vel_reduced_P_MnI"].T
    vel_reduced_P_MgII   = QS_data["vel_reduced_P_MgII"].T 

plot_velocity_map(vel_reduced_P_MnI.T, vmin=-2, vmax=2, aspect="auto", 
                  title="Mn I 2801.5 $\AA$ Doppler velocity ")


plot_velocity_map(vel_reduced_P_MgII.T, vmin=-7, vmax=7, aspect="auto", 
                  title="Mg II h 2v Doppler velocity ")
(WCT, angles, corr_sig_all, 
 freq, coi_mask)  = calc_phase_diff_full(vel_reduced_P_MnI[10:120, 100:-50], 
                                         vel_reduced_P_MgII[10:120, 100:-50], 
                                         dt=2.2, t0=0, 
                                         d0=72, 
                                         d1=123)

angles_average = [angle_mean(el.flatten()) for el in 
                  angles * coi_mask[:, :, None]]
angles_average = np.array(angles_average)

angles_masked = angles * (cor_sig_all > 1) 
angles_masked *= coi_mask[:, :, None]


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
pl.title("Phase Difference Mn I - Mg II velocity")
pl.grid()
pl.legend()
pl.ylabel("Phase difference [deg]")