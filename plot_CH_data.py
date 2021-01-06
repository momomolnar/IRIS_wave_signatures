#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:35:45 2020

@author: molnarad
"""

import numpy as np
from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D



file_name = ["../IRIS_data/CH_data/CH_vel_mgk_mu=0.99.sav",
             "../IRIS_data/CH_data/CH_vel_mgk_mu=0.93.sav",
             "../IRIS_data/CH_data/CH_vel_mgk_mu=0.83.sav",
             "../IRIS_data/CH_data/CH_vel_mgk_mu=0.75.sav",
             "../IRIS_data/CH_data/CH_vel_mgk_mu=0.65.sav",
             "../IRIS_data/CH_data/CH_vel_mgk_mu=0.34.sav",]
file_index = 1
file_name = file_name[file_index]
mu_angle = file_name[-8  :-4]
print(f"The mu angle is {mu_angle}")
d= "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Meetings/Mettings_with_Steve/2020_11_19/"

cadence = [9.47, 16.39, 9.33, 9.33, 16.21, 9.57]
dt      = cadence[file_index]


bp, rp, lc = load_velocity_mg2(file_name)
#Do the 2v lines first 

dd_param = 1
asp= 0.075/dd_param
asp_pxx = 5/dd_param*3

k_2v_v = filter_velocity_field(bp[20:-20, 10:-10, 0, 0], dd=dd_param)
plot_velocity_map(k_2v_v, title="CH k2v velocity @ mu="+mu_angle, 
                  aspect=asp, d=d, cadence = dt)
plot_intensity_map(average_velmap_slitwise(np.swapaxes(bp[10:-10, 10:-10, 1, 0], 
                                                       0, 1), 1),
                   title="k2v intensity @ mu="+mu_angle, 
                   aspect=asp, 
                   vmin=10, vmax=300, d=d, cadence = dt)
freq, Pxx_k_2v_v = calc_Pxx_velmap(k_2v_v, fsi=1/dt)
plot_Pxx_2D(freq, Pxx_k_2v_v, title="CH PSD k2v intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=d)