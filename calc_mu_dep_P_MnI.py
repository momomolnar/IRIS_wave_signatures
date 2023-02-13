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
from IRIS_lib import obs_details
import pandas as pd

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Plage_data/"
filename = ["MnI_2801.9_0.22.sav",
            "MnI_2801.9_0.39.sav",
            # "MnI_2801.9_0.44.sav",
            "MnI_2801.9_0.48.sav",
            # "MnI_2801.9_0.70.sav",
            "MnI_2801.9_0.87.sav",
            # "MnI_2801.9_1.00.sav",]
            "MnI_2801.9_1.000.sav",
            "MnI_2801.9_mu=0.10.sav"]

cadence = [17, 17, 4, 5, 9, 9, 5, 9, 16.8]
num_files = len(filename)
mu_angles = np.zeros(num_files)
v_rms = np.zeros(num_files)

obs_xy_angle = [[933, 83, 0],
                [881, 75, 0],
                [660, -273, 0],
                [425, -190, 0],
                [38 , 81, 0],
                [62, 59, 0],
                [961, 117]]

lim_vel = [[220, 325], 
           [250, 400], 
           [100, 250], 
           # [300, 450], 
           [250, 550], 
           #[300, 650],
           [500, 700],
           [100, -100],
           ]

observations_reduced_df = pd.DataFrame(columns=["mu", "v_rms"],)

Pxx_noise_ave = np.zeros(num_files)


freq_int = [3e-3, 20e-3]
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
                            num_pixels=num_pixels, 
                            R_sol=975)
    obs_props.mu_pixel()
    
    vel_reduced = filter_velocity_field(velocities,  
                                        dt=5, dx=5, dd=1, degree=1)
    v_lim = 2

    plot_velocity_map(vel_reduced, vmin=-1*v_lim, vmax=v_lim, aspect=0.1, 
                      title="CH Mn I Velocity " + mu_angle)

    #im1 = pl.imshow(vel_reduced.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
    #pl.colorbar(im1)
    #pl.title("CH Mn I Velocity " + mu_angle)
    #pl.show() 
    
    
    

    freq, Pxx     = signal.periodogram(vel_reduced[lim_vel[el1][0]:lim_vel[el1][1],
                                                   :], fs=1/cadence[el1],
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
                                                
    plot_Pxx_2D(freq, Pxx_denoise[:, :], aspect=1/0.3/0.3)

    # Add noise estimate
    
    ## ADD THE DISTRIBUTION in MU 
    ## as a Pandas dataframe and then plot as a distribution over Mu
    for el in zip(np.flipud(obs_props.mu)[lim_vel[el1][0]:lim_vel[el1][1]], 
                  v_fluc_total):
        observations_reduced_df = observations_reduced_df.append({"mu":el[0], 
                                                                  "v_rms":el[1]},
                                                                 ignore_index=True) 
    
    # Add noise estimate
    
    
   
    print(Pxx_spatial_average)
    v_rms[el1] = np.sum(Pxx_spatial_average[int(freq_lim[0]):int(freq_lim[1])]) * freq[1]
    print(f"V_RMS_{el1} is {v_rms[el1]} $\pm$ {Pxx_noise_ave[el1]}.") 

data_dir = ("/Users/molnarad/CU_Boulder/Work/Chromospheric_business/"
            + "IRIS_waves/IRIS_data/Sample_datasets_CLV/")

observations_reduced_df.to_csv(data_dir + "P_MnI_data.pd")

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
pl.ylabel("Vel_rms [km/s]")
pl.xlabel("$\mu^2$")