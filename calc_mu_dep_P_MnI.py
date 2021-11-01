#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:57:44 2021

Calculate the mu_angle dependence of <v^2> for Plage 

@author: molnarad
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from scipy import signal
from IRIS_lib import plot_velocity_map, filter_velocity_field, plot_Pxx_2D

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Plage_data/"
filename = ["MnI_2801.9_0.22.sav",
            "MnI_2801.9_0.39.sav",
            "MnI_2801.9_0.44.sav",
            "MnI_2801.9_0.48.sav",
            "MnI_2801.9_0.70.sav",
            "MnI_2801.9_0.87.sav",
            "MnI_2801.9_1.00.sav",
            "MnI_2801.9_1.000.sav"]

cadence = [17, 17, 4, 5, 9, 9, 5, 9]
num_files = len(filename)
mu_angles = np.zeros(num_files)
v_rms = np.zeros(num_files)

freq_int = [5e-3, 15e-3]

for el1 in range(num_files):
    aa = io.readsav(d+filename[el1])
    
    mu_angle = filename[el1][-8:-4]
    if mu_angle == ".000":
        mu_angle = "1.01"
    mu_angles[el1] = mu_angle
    
    fit_params = aa["fit_params"]
    velocities = fit_params[1, :, :]
    rest_wave = np.median(velocities)
    velocities = (velocities - rest_wave)*3e5/2801.9
    vel_reduced = filter_velocity_field(velocities,  
                                        dt=5, dx=5, dd=1, degree=1)
    v_lim = 2

    plot_velocity_map(vel_reduced.T, vmin=-1*v_lim, vmax=v_lim, aspect=0.1, 
                      title="CH Mn I Velocity " + mu_angle)

    #im1 = pl.imshow(vel_reduced.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
    #pl.colorbar(im1)
    #pl.title("CH Mn I Velocity " + mu_angle)
    #pl.show() 
    
    lim_vel = [50, -150]
    
    
    freq, _ = signal.periodogram(vel_reduced[:, 0], fs=1/cadence[el1])
    Pxx     = [signal.periodogram(el, fs=1/cadence[el1])[1] for el in 
               vel_reduced[:, lim_vel[0]:lim_vel[1]].T]
    Pxx     = np.array(Pxx)
    
    plot_Pxx_2D(freq, Pxx, aspect=1/0.3/0.3)
    
    freq_lim = (freq_int // freq[1])
    
    Pxx_spatial_average = np.nanquantile(Pxx, 0.5, axis=0)
    
    # Add noise estimate
    
    Pxx_noise = np.nanquantile(Pxx_spatial_average[-60:-30], 0.5)
    Pxx_spatial_average -= Pxx_noise 
    
    print(Pxx_spatial_average)
    v_rms[el1] = np.sum(Pxx_spatial_average[int(freq_lim[0]):int(freq_lim[1])]) * freq[1]
    print(f"V_RMS_{el1} is {v_rms[el1]}") 
    
pl.plot(mu_angles**2, v_rms, 'b.--')
pl.xlabel("mu^2")
pl.ylabel("V$_{rms}$ [(km/s)$^2$]")
pl.show()