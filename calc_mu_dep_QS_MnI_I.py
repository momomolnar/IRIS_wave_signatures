#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:31:47 2021

@author: molnarad
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from scipy import signal
from IRIS_lib import plot_velocity_map, filter_velocity_field, plot_Pxx_2D

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/QS_data/"
filename = ["MnI_2801.9_mu=0.2.sav",
            "MnI_2801.9_mu=0.4.sav",
            "MnI_2801.9_mu=0.6.sav",
            "MnI_2801.9_mu=0.8.sav",
            "MnI_2801.9_mu=1.0.sav"]

cadence = [16.7, 16.7, 16.7, 16.7, 16.7, 16]
num_files = len(filename)
mu_angles = np.zeros(num_files)
v_rms = np.zeros(num_files)

freq_int = [5e-3, 15e-3]

Pxx_noise_ave = np.zeros(num_files)

for el1 in range(num_files):
    aa = io.readsav(d+filename[el1])
    
    mu_angle = filename[el1][-7:-4]
    mu_angles[el1] = mu_angle

    fit_params = aa["fit_params"]
    velocities = fit_params[0, :, :]
    I_0 = np.median(velocities, axis=1)
    velocities = (velocities - I_0[:, None])
    vel_reduced = filter_velocity_field(velocities,  
                                        dt=100, dx=100, dd=1, degree=1)
    v_lim = 0.5

    plot_velocity_map(vel_reduced.T/I_0[2:-2, None], vmin=-1*v_lim, vmax=v_lim, aspect=0.1, 
                      title="CH Mn I Velocity " + mu_angle)

    #im1 = pl.imshow(vel_reduced.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
    #pl.colorbar(im1)
    #pl.title("CH Mn I Velocity " + mu_angle)
    #pl.show() 
    
    lim_vel = [50, -50]
    
    
    freq, _ = signal.periodogram(vel_reduced[:, 0], fs=1/cadence[el1])
    Pxx     = [signal.periodogram(el, fs=1/cadence[el1])[1] for el in 
               (vel_reduced[:, lim_vel[0]:lim_vel[1]].T/I_0[lim_vel[0]+2:lim_vel[1]-2, None])]
    Pxx     = np.array(Pxx)
       
    freq_lim = (freq_int // freq[1])
    Pxx_noise = np.nanquantile(Pxx[:, -30:], 0.5, axis=1)
    Pxx_denoise = Pxx - Pxx_noise[:, None]
    Pxx_noise_ave[el1] = (np.nanquantile(Pxx_noise[50:-50], 0.5) 
                          * freq[1] * (freq_lim[1] - freq_lim[0])) / np.sqrt(100)
    Pxx_spatial_average = np.nanquantile(Pxx_denoise[40:-40], 0.5, axis=0) 
                           
                         
    plot_Pxx_2D(freq, Pxx_denoise[40:-40], aspect=1/0.3/0.3)

    # Add noise estimate
    
    
   
    print(Pxx_spatial_average)
    v_rms[el1] = np.sum(Pxx_spatial_average[int(freq_lim[0]):int(freq_lim[1])]) * freq[1]
    print(f"V_RMS_{el1} is {v_rms[el1]} $\pm$ {Pxx_noise_ave[el1]}.") 

pl.figure(figsize=(4,3), dpi=250)
pl.errorbar(mu_angles**2, v_rms, fmt='.', yerr=Pxx_noise_ave, label="Data")
coeffs = np.polyfit(mu_angles**2, v_rms, deg=1, w=1/Pxx_noise_ave)
x = np.linspace(0, 1, num=100)
pl.plot(x**2, (coeffs[0]* (x**2) + coeffs[1]), '--', label="Model")
pl.xlabel("$\mu^2$")
pl.grid(alpha=0.5)
pl.legend()
pl.title("QS Mn I $\langle \delta I / I \\rangle$ $\mu$-dependence") 
pl.ylabel("($\delta I / I$)$^2$ variation")
#pl.ylim(0.0, 0.3)
pl.show()