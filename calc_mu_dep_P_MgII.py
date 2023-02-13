
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
from IRIS_lib import plot_intensity_map, obs_details, running_av_mu
import pandas as pd

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Plage_data/"
filename = ["iris_l2_20160101_150128_3690091404_raster_t000_r00000__0.22_mg2_vel.sav",
            "iris_l2_20160101_020028_3690091404_raster_t000_r00000__0.39_mg2_vel.sav",
            #"iris_l2_20130820_123133_4004255648_raster_t000_r00000__0.44_mg2_vel.sav",
            "iris_l2_20131117_194238_3884008102_raster_t000_r00000__0.67_mg2_vel.sav",
            #"iris_l2_20200725_000137_3620508703_raster_t000_r00000__0.70_mg2_vel.sav",
            "iris_l2_20131213_070938_3820009403_raster_t000_r00000__0.87_mg2_vel.sav",
            # "iris_l2_20140918_080253_3820257453_raster_t000_r00000__1.00_mg2_vel.sav",
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


obs_xy_angle = [[933, 83, 0],
                [881, 75, 0],
                [660, -273, 0],
                [425, -190, 0],
                # [38 , 81, 0],
                [62, 59, 0]]

lim_vel = [[220, 325], 
           [250, 400], 
           [100, 250], 
           # [300, 450], 
           [250, 550], 
           # [300, 650],
           [500, 700]]


freq_int = [3e-3, 20e-3]

Pxx_noise_ave = np.zeros(num_files)

observations_reduced_df = pd.DataFrame(columns=["mu", "v_rms", "Line"],)

line_list = ["k2v", "k3", "k2r"]

for el1 in range(num_files):
    for line in range(len(line_list)):
    #plot intensity 
    #pick only top 50% of intensity 
    # compare intensity fluctuations with velocity fluctuations -- if it's alfvenic waves, no correlation!
    
        print(filename[el1])
 
        mu_angle = filename[el1][-16:-12]

        mu_angles[el1] = mu_angle

        aa = load_velocity_mg2(d + filename[el1])
        
        v = aa[line]
    
        lc_temp = np.nanmedian(v[t_limits[0]:t_limits[1], 
                                 slit_limits[0]:slit_limits[1], 0, :], 
                               axis=-1).T
        
        lc_temp = np.nanmedian(v[t_limits[0]:t_limits[1], 
                                 slit_limits[0]:slit_limits[1], 0, :], 
                               axis=-1).T
    
        num_pixels = lc_temp.shape[0]
    
        obs_props = obs_details(obs_xy_angle[el1][0], obs_xy_angle[el1][1], 
                                obs_xy_angle[el1][2],
                                num_pixels=num_pixels)
        obs_props.mu_pixel()

    
        vel_reduced = filter_velocity_field(lc_temp, dt=7, 
                                            dx=7, dd=3, degree=1)
        v_lim = 7

        plot_velocity_map(vel_reduced, vmin=-1*v_lim, vmax=v_lim, aspect=0.1, 
                      title="CH Mg II Velocity " + mu_angle)
        bp_new = np.swapaxes(v[t_limits[0]:t_limits[1], 
                               slit_limits[0]:slit_limits[1], 1, 
                               line_index], 0, 1)
        plot_intensity_map(bp_new,
                           title="CH k2v intensity @ mu="+mu_angle, 
                           aspect=0.1, 
                           vmin=150, #np.amin(bp_new[~(np.isnan(bp_new))])*1.2,
                           vmax=1000, #np.amax(bp_new[~(np.isnan(bp_new))])*0.8,
                           d=d, cadence = cadence[el1])

    #im1 = pl.imshow(vel_reduced.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
    #pl.colorbar(im1)
    #pl.title("CH Mn I Velocity " + mu_angle)
    #pl.show() 
    

    
    
        freq, Pxx     = signal.periodogram(vel_reduced[lim_vel[el1][0]:lim_vel[el1][1],
                                                       10:-10], 
                                           fs=1/cadence[el1], axis=-1)
    
        plot_Pxx_2D(freq, Pxx, aspect=1/0.3/0.3, title="Pxx raw")
    
        freq_lim = (freq_int // freq[1])
    
        Pxx_noise = np.nanmedian(Pxx[:, -30:], axis=-1)
    
        Pxx_denoise = Pxx - Pxx_noise[:, None]
    
        v_fluc_total = (np.sum(Pxx_denoise[:, int(freq_lim[0]):int(freq_lim[1])],
                              axis=-1) 
                        * freq[1])
        Pxx_noise_ave[el1] = (np.nanquantile(Pxx_noise[50:-50], 0.5) 
                          * freq[1] * (freq_lim[1] - freq_lim[0]))/np.sqrt(len(Pxx_noise[50:-50]))
        Pxx_spatial_average = np.nanquantile(Pxx_denoise[40:-40], 0.5, axis=0) 
                                                
        plot_Pxx_2D(freq, Pxx_denoise[40:-40, :], aspect=1/0.3/0.3, 
                    title="Pxx noise subtracted")

    # Add noise estimate
    
    ## ADD THE DISTRIBUTION in MU 
    ## as a Pandas dataframe and then plot as a distribution over Mu
        for el in zip(obs_props.mu[lim_vel[el1][0]:lim_vel[el1][1]], 
                      v_fluc_total):
            observations_reduced_df = observations_reduced_df.append({"mu":el[0], 
                                                                      "v_rms":el[1], 
                                                                      "Line":line_list[line]},
                                                                      ignore_index=True) 
    
    #print(Pxx_spatial_average)
        v_rms[el1] = np.sum(Pxx_spatial_average[int(freq_lim[0]):int(freq_lim[1])]) * freq[1]
        print(f"V_RMS_{el1} is {v_rms[el1]} $\pm$ {Pxx_noise_ave[el1]}.") 

data_dir = ("/Users/molnarad/CU_Boulder/Work/Chromospheric_business/"
            + "IRIS_waves/IRIS_data/Sample_datasets_CLV/")

observations_reduced_df.to_csv(data_dir + "P_MgII_data.pd")


pd = observations_reduced_df 

k3 = pd.loc[pd["Line"] == "k3"]
k2v = pd.loc[pd["Line"] == "k2v"]
k2r = pd.loc[pd["Line"] == "k2r"]

mu_av = np.linspace(0, 1, num=20)
k3_av = running_av_mu(k3, mu_av)
k2r_av = running_av_mu(k2r, mu_av)
k2v_av = running_av_mu(k2v, mu_av)

alpha=0.05

pl.plot(k3.mu, k3.v_rms, 'ro', alpha=alpha)
pl.plot(k3_av[:, 0], k3_av[:, 1], 'r.--', label="k3")

pl.plot(k2r.mu, k2r.v_rms, 'go', alpha=alpha)
pl.plot(k2r_av[:, 0], k2r_av[:, 1], 'g.--', label="k2r")

pl.plot(k2v.mu, k2v.v_rms, 'bo', alpha=alpha)
pl.plot(k2v_av[:, 0], k2v_av[:, 1], 'b.--', label="k2v")

pl.legend()
pl.ylim(0, 20)
pl.ylabel("$\langle V^2 \\rangle$ [(km/s)$^2$]")
pl.xlabel("$\mu^2$")
pl.title("CH MgII h/k line $\langle V^2 \\rangle$")
pl.show()