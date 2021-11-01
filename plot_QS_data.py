#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:37:32 2021

Plot the QS data in

@author: molnarad
"""

import numpy as np
from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D

file_name = ["../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.2.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.4.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.6.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.8.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=1.0.sav",
             "../IRIS_data/QS_data/iris_l2_20131116_073345_3803010103_raster_t000_r00000_mg2_vel_mu=1.0.sav",
             "../IRIS_data/QS_data/iris_l2_20140201_104147_4004257547_raster_t000_r00000_mu06_mg2_vel.sav",
             "../IRIS_data/QS_data/iris_l2_20140201_104147_4004257547_raster_t000_r00000_mu06_mg2_vel.sav"]

file_index = 2
file_name = file_name[file_index]
mu_angle = file_name[-7:-4]
print(f"The mu angle is {mu_angle}")
d= "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Meetings/Meetings_with_Steve/2021_02_25/"

cadence = [16.7, 16.7, 16.7, 16.7, 16.7, 16.7, 16.7, 16.7, 5]
dt      = cadence[file_index]


bp, rp, lc = load_velocity_mg2(file_name)
#Do the 2v lines first 

dd_param = 1
slit_av_param = 1


asp= "auto" # 0.075/dd_param/2
asp_pxx = "auto" # 5/dd_param*5

t_limits = [1, -2]
slit_limits = [20,-40]

# bp/lc/rp = [Nt, Nslit, v[0]/I[1], line h[0]/k[1]]

line_index = 0
line_index1= 1 

k_2v_v = filter_velocity_field(bp[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, line_index], 
                               dt=12, dx=12, dd=3, degree=1)

h_2v_v = filter_velocity_field(bp[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, line_index1], 
                               dt=12, dx=12, dd=3, degree=1)
average_map_2v = (k_2v_v+h_2v_v)/2

plot_velocity_map((k_2v_v+h_2v_v)/2, title="QS k2v velocity @ mu="+mu_angle, 
                  aspect=asp, d=d, cadence = dt)
bp_new = np.swapaxes(bp[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1],
                        1, line_index], 0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param),
                   title="QS k2v intensity @ mu="+mu_angle, 
                   aspect=asp, 
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.2,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.5,
                   d=d, cadence = dt)
freq, Pxx_k_2v_v = calc_Pxx_velmap(average_map_2v, fsi=1/dt)
Pxx_k_2v_v -= (np.median(Pxx_k_2v_v[:, -25:-1], axis=1))[:, None]
plot_Pxx_2D(freq, Pxx_k_2v_v, title="QS PSD k2v intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=d, remove_noise=False)

k_3_v = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                 slit_limits[0]:slit_limits[1], 0, line_index],
                              dt=12, dx=12, dd=3, degree=1)
h_3_v = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                 slit_limits[0]:slit_limits[1], 0, line_index1],
                              dt=12, dx=12, dd=3, degree=1)

average_map_3 = 0.5*(k_3_v + h_3_v)

plot_velocity_map(average_map_3, title="QS k3 velocity @ mu="+mu_angle, 
                  aspect=asp, d=d, cadence=dt, vmin=-7.5, vmax=7.5)
bp_new = np.swapaxes(lc[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 
                        1, line_index], 
                     0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param), 
                   title="QS k3 intensity @ mu="+mu_angle, aspect=asp, 
                   vmin=250,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.3,
                   d=d, cadence=dt)
freq, Pxx_k_3_v = calc_Pxx_velmap(average_map_3, fsi=1/dt)
Pxx_k_3_v -= (np.median(Pxx_k_3_v[:, -25:-1], axis=1))[:, None]
plot_Pxx_2D(freq, Pxx_k_3_v, title="QS k3 PSD velocity @ mu="+mu_angle,
            aspect=asp_pxx, d=d, vmaxa=.5, remove_noise=False)