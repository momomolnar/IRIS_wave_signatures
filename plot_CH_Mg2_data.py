#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:14:54 2021

Show the CH Mg2 data

@author: molnarad
"""
import numpy as np
from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D


dir_ = "../IRIS_data/CH_data/"

file_name = ["iris_l2_20140511_051421_3820259553_raster_t000_r00000__0.39_mg2_vel.sav",
             "iris_l2_20161107_025410_3620109603_raster_t000_r00000__0.82_mg2_vel.sav",
             "iris_l2_20160321_210428_3601110003_raster_t000_r00000__0.41_mg2_vel.sav",
             "iris_l2_20161008_194709_3620259603_raster_t000_r00000__0.25_mg2_vel.sav",
             "iris_l2_20170321_195128_3603111603_raster_t000_r00000__0.92_mg2_vel.sav",
             "iris_l2_20161025_111933_3620109603_raster_t000_r00000__0.74_mg2_vel.sav"]


file_index = 0
file_name = file_name[file_index]
mu_angle = file_name[-16:-12]
print(f"The mu angle is {mu_angle}")

cadence = [9.6, 16.39, 9.33, 9.33, 16.21, 9.57]
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
plot_velocity_map(k_2v_v, title="CH k2v velocity @ mu="+mu_angle, 
                  aspect=asp, d=dir_, cadence = dt)
bp_new = np.swapaxes(bp[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 1, line_index], 0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param),
                   title="CH k2v intensity @ mu="+mu_angle, 
                   aspect=asp, 
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.2,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.5,
                   d=dir_, cadence = dt)
freq, Pxx_k_2v_v = calc_Pxx_velmap(k_2v_v, fsi=1/dt)
plot_Pxx_2D(freq, Pxx_k_2v_v, title="CH PSD k2v intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=dir_, remove_noise=False)

k_3_v = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                 slit_limits[0]:slit_limits[1], 0, line_index],
                              dt=7, dx=7, dd=5, degree=2)
plot_velocity_map(k_3_v, title="CH k3 velocity @ mu="+mu_angle, 
                  aspect=asp, d=dir_, cadence=dt, vmin=-9, vmax=9)
bp_new = np.swapaxes(lc[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 
                        1, line_index], 
                     0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param), 
                   title="CH k3 intensity @ mu="+mu_angle, aspect=asp, 
                   vmin=250,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.3,
                   d=dir_, cadence=dt)
freq, Pxx_k_3_v = calc_Pxx_velmap(k_3_v, fsi=1/dt)
plot_Pxx_2D(freq, Pxx_k_3_v, title="CH PSD k3 intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=dir_, vmaxa=.5, remove_noise=False)