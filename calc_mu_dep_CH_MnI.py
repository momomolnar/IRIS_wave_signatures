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
from IRIS_lib import obs_details
from IRIS_lib import running_av_mu

import pandas as pd

        

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/CH_data/"
filename = ["MnI_2801.9_0.25.sav",
            "MnI_2801.9_0.39.sav",
            "MnI_2801.9_0.64.sav",
            "MnI_2801.9_0.74.sav",
            "MnI_2801.9_0.82.sav",
            "MnI_2801.9_0.92.sav"]

obs_xy_angle = [[89, -892, 0], 
                [401, -614, 0],
                [-458,-209, 0],
                [629, 350, 0],
                [-494, -394, 0],
                [351, 95, 0]]
                           

cadence = [9.6, 9.6, 16.1, 9.6, 9.6, 16]
num_files = len(filename)
mu_angles = np.zeros(num_files)
v_rms = np.zeros(num_files)

freq_int = [5e-3, 25e-3]

Pxx_noise_ave = np.zeros(num_files)

observations_reduced_df = pd.DataFrame(columns=["mu", "v_rms"],)

for el1 in range(num_files):
    print(filename[el1])
    aa = io.readsav(d+filename[el1])
    
    mu_angle = filename[el1][-8:-4]
    mu_angles[el1] = mu_angle

    fit_params = aa["fit_params"]
    velocities = fit_params[1, :, :]
    rest_wave = np.median(velocities)
    velocities = (velocities - rest_wave)*3e5/2801.9
    
    num_pixels = velocities.shape[0]
    
    obs_props = obs_details(obs_xy_angle[el1][0], obs_xy_angle[el1][1], 
                            obs_xy_angle[el1][2],
                            num_pixels=num_pixels)
    obs_props.mu_pixel()
    
    vel_reduced = filter_velocity_field(velocities,  
                                        dt=5, dx=5, dd=1, degree=1)
    v_lim = 2

    plot_velocity_map(vel_reduced, vmin=-1*v_lim, vmax=v_lim, aspect=0.015, 
                      title="CH Mn I Velocity " + mu_angle)

    #im1 = pl.imshow(vel_reduced.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
    #pl.colorbar(im1)
    #pl.title("CH Mn I Velocity " + mu_angle)
    #pl.show() 
    
    
    

    freq, Pxx     = signal.periodogram(vel_reduced, fs=1/cadence[el1],
                                       axis=-1)
    
    plot_Pxx_2D(freq, Pxx, aspect=1/0.3/0.3)
    
    freq_lim = (freq_int // freq[1])
    
    Pxx_noise = np.nanmedian(Pxx[:, -30:], axis=-1)
    
    Pxx_denoise = Pxx - Pxx_noise[:, None]
    
    v_fluc_total = (np.sum(Pxx_denoise[:, int(freq_lim[0]):int(freq_lim[1])],
                           axis=-1) 
                    * freq[1])
    Pxx_noise_ave[el1] = (np.nanquantile(Pxx_noise[50:-50], 0.5) 
                          * freq[1] * (freq_lim[1] - freq_lim[0]))/np.sqrt(len(Pxx_noise[50:-50]))
    Pxx_spatial_average = np.nanquantile(Pxx_denoise[40:-40], 0.5, axis=0) 
                                                
    plot_Pxx_2D(freq, Pxx_denoise[40:-40, :], aspect=1/0.3/0.3)

    # Add noise estimate
    
    ## ADD THE DISTRIBUTION in MU 
    ## as a Pandas dataframe and then plot as a distribution over Mu
    for el in zip(obs_props.mu, v_fluc_total):
        observations_reduced_df = observations_reduced_df.append({"mu":el[0], 
                                                                  "v_rms":el[1]},
                                                                 ignore_index=True) 
    
    # Add noise estimate
    
    
   
    print(Pxx_spatial_average)
    v_rms[el1] = np.sum(Pxx_spatial_average[int(freq_lim[0]):int(freq_lim[1])]) * freq[1]
    print(f"V_RMS_{el1} is {v_rms[el1]} $\pm$ {Pxx_noise_ave[el1]}.") 

data_dir = ("/Users/molnarad/CU_Boulder/Work/Chromospheric_business/"
            + "IRIS_waves/IRIS_data/Sample_datasets_CLV/")

observations_reduced_df.to_csv(data_dir + "CH_MnI_data.pd")

pl.figure(figsize=(4,3), dpi=250)
pl.errorbar(mu_angles**2, v_rms, fmt='.', yerr=Pxx_noise_ave, label="Data")
coeffs = np.polyfit(mu_angles**2, v_rms, deg=1, w=1/Pxx_noise_ave)
x = np.linspace(0, 1, num=100)
pl.plot(x**2, (coeffs[0]* (x**2) + coeffs[1]), '--', label="Model")
pl.xlabel("$\mu^2$")
pl.grid(alpha=0.5)
pl.legend()
pl.title("CH Mn I $\langle v^2 \\rangle$ $\mu$-dependence") 
pl.ylabel("V$^2_{RMS}$ [(km/s)$^2$]")
pl.ylim(0, 0.35)
pl.show() 

pl.plot(observations_reduced_df.mu**2, 
        observations_reduced_df.v_rms, 'r.', alpha=0.1)
pl.ylim(0, 0.4)
pl.xlim(0, 1) 
pl.title("CH Mn I $\langle v^2 \\rangle$ $\mu$-dependence") 
pl.ylabel("Vel_rms [km/s]")
pl.xlabel("$\mu^2$")