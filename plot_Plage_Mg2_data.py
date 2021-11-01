#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:37:31 2021

Plot the Mg2 plage data 

@author: molnarad
"""

import numpy as np
from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D


dir_ = "../IRIS_data/Plage_data/"

file_name = ["iris_l2_20130820_123133_4004255648_raster_t000_r00000__0.44_mg2_vel.sav",
             "iris_l2_20131117_194238_3884008102_raster_t000_r00000__0.67_mg2_vel.sav",
             "iris_l2_20131213_070938_3820009403_raster_t000_r00000__0.87_mg2_vel.sav",
             "iris_l2_20140918_080253_3820257453_raster_t000_r00000__1.00_mg2_vel.sav",
             "iris_l2_20140918_101908_3820259153_raster_t000_r00000__1.00_mg2_vel.sav",
             "iris_l2_20160101_020028_3690091404_raster_t000_r00000__0.39_mg2_vel.sav",
             "iris_l2_20160101_150128_3690091404_raster_t000_r00000__0.22_mg2_vel.sav",
             "iris_l2_20200725_000137_3620508703_raster_t000_r00000__0.70_mg2_vel.sav"]

file_index = 6
file_name = file_name[file_index]
mu_angle = file_name[-16:-12]
print(f"The mu angle is {mu_angle}")

cadence = [4, 9, 9, 5, 9, 17, 17, 9]
dt      = cadence[file_index]


bp, rp, lc = load_velocity_mg2(dir_ + file_name)
#Do the 2v lines first 

dd_param = 1
slit_av_param = 1


asp= "auto" # 0.075/dd_param/2
asp_pxx = "auto" # 5/dd_param*5

t_limits = [1, -2]
slit_limits = [10,-100]

line_index = 1 

k_2v_v = filter_velocity_field(bp[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, line_index], 
                               dt=7, dx=7, dd=3, degree=1)
plot_velocity_map(k_2v_v, title="Plage k2v velocity @ mu="+mu_angle, 
                  aspect=asp, d=dir_, cadence = dt)
bp_new = np.swapaxes(bp[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 1, line_index], 0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param),
                   title="Plage k2v intensity @ mu="+mu_angle, 
                   aspect=asp, 
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.2,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*0.8,
                   d=dir_, cadence = dt)
freq, Pxx_k_2v_v = calc_Pxx_velmap(k_2v_v, fsi=1/dt)
plot_Pxx_2D(freq, Pxx_k_2v_v, title="Plage PSD k2v intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=dir_, remove_noise=False)

k_3_v = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                 slit_limits[0]:slit_limits[1], 0, line_index],
                              dt=7, dx=7, dd=5, degree=2)
plot_velocity_map(k_3_v, title="Plage k3 velocity @ mu="+mu_angle, 
                  aspect=asp, d=dir_, cadence=dt, vmin=-9, vmax=9)
bp_new = np.swapaxes(lc[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 
                        1, line_index], 
                     0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param), 
                   title="Plage k3 intensity @ mu="+mu_angle, aspect=asp, 
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.2,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.8,
                   d=dir_, cadence=dt)
freq, Pxx_k_3_v = calc_Pxx_velmap(k_3_v, fsi=1/dt)
plot_Pxx_2D(freq, Pxx_k_3_v, title="Plage PSD k3 intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=dir_, vmaxa=.5, remove_noise=False)
