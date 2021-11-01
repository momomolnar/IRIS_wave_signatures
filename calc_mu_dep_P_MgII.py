
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
from IRIS_lib import filter_velocity_field, load_velocity_mg2
from IRIS_lib import plot_intensity_map

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Plage_data/"
filename = ["iris_l2_20160101_150128_3690091404_raster_t000_r00000__0.22_mg2_vel.sav",
            "iris_l2_20160101_020028_3690091404_raster_t000_r00000__0.39_mg2_vel.sav",
            #"iris_l2_20130820_123133_4004255648_raster_t000_r00000__0.44_mg2_vel.sav",
            "iris_l2_20131117_194238_3884008102_raster_t000_r00000__0.67_mg2_vel.sav",
            "iris_l2_20200725_000137_3620508703_raster_t000_r00000__0.70_mg2_vel.sav",
            "iris_l2_20131213_070938_3820009403_raster_t000_r00000__0.87_mg2_vel.sav",
            "iris_l2_20140918_080253_3820257453_raster_t000_r00000__1.00_mg2_vel.sav",
            "iris_l2_20140918_101908_3820259153_raster_t000_r00000__1.00_mg2_vel.sav"]
cadence = [17, 17, 5, 9, 9, 5, 9]
#cadence = [17, 17, 4, 5, 9, 9, 5, 9] # without the mu=0.44 dataset 
num_files = len(filename)
mu_angles = np.zeros(num_files)
v_rms = np.zeros(num_files)
v_rms_noise = np.zeros(num_files)
t_limits = [1, -2]
line_index = 0
slit_limits = [0, -1]

freq_int = [5e-3, 15e-3]

lim_vel = [[220, 325], [250, 400], [100, 250], [300, 450], [250, 550], [300, 650],
           [500, 700]]

for el1 in range(num_files):
    #plot intensity 
    #pick only top 50% of intensity 
    # compare intensity fluctuations with velocity fluctuations -- if it's alfvenic waves, no correlation!
    mu_angle = filename[el1][-16:-12]

    mu_angles[el1] = mu_angle
    bp, rp, lc = load_velocity_mg2(d + filename[el1])
    vel_reduced = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, 1], 
                               dt=7, dx=7, dd=3, degree=1)
    v_lim = 7

    plot_velocity_map(vel_reduced, vmin=-1*v_lim, vmax=v_lim, aspect=0.1, 
                      title="Plage Mg II Velocity " + mu_angle)
    bp_new = np.swapaxes(lc[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 1, line_index], 0, 1)
    plot_intensity_map(bp_new,
                   title="Plage k2v intensity @ mu="+mu_angle, 
                   aspect=0.1, 
                   vmin=150, #np.amin(bp_new[~(np.isnan(bp_new))])*1.2,
                   vmax=1000, #np.amax(bp_new[~(np.isnan(bp_new))])*0.8,
                   d=d, cadence = cadence[el1])

    #im1 = pl.imshow(vel_reduced.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
    #pl.colorbar(im1)
    #pl.title("CH Mn I Velocity " + mu_angle)
    #pl.show() 
    

    
    
    freq, _ = signal.periodogram(vel_reduced[:, 0], fs=1/cadence[el1])
    Pxx     = [signal.periodogram(el, fs=1/cadence[el1])[1] for el in 
               vel_reduced[lim_vel[el1][0]:lim_vel[el1][1], :]]
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
pl.show()