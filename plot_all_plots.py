#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 00:06:33 2020

@author: molnarad
"""

import numpy as np
from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D



file_name = ["../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.2.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.4.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.8.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=1.0.sav"]

d= "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Meetings/Mettings_with_Steve/2021_01_08/"

file_name = file_name[0]
mu_angle = file_name[-7:-4]
print(f"The mu angle is {mu_angle}")

bp, rp, lc = load_velocity_mg2(file_name)
#Do the 2v lines first 

dd_param = 2
slit_av_param = 3
asp= 0.075*3/dd_param
asp_pxx = 5/dd_param*6

t_limits = [1, -2]
slit_limits = [10, -100]

k_2v_v = filter_velocity_field(bp[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, 0], dd=dd_param)
plot_velocity_map(k_2v_v, title="k2v velocity @ mu="+mu_angle, 
                  aspect=asp, d=d)
bp_new = np.swapaxes(bp[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 1, 0], 0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param), 
                   title="k2v intensity @ mu="+mu_angle, 
                   aspect=asp,
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.3,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.5,
                   d=d)
freq, Pxx_k_2v_v = calc_Pxx_velmap(k_2v_v)
plot_Pxx_2D(freq, Pxx_k_2v_v, title="PSD k2v intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=d)

h_2v_v = filter_velocity_field(bp[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, 1], dd=dd_param)
plot_velocity_map(h_2v_v, title="h2v velocity @ mu="+mu_angle, aspect=asp, d=d)
bp_new = np.swapaxes(bp[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 1, 1], 
                     0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param),
                   title="h2v intensity @ mu="+mu_angle, aspect=asp, 
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.3,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.5,
                   d=d)
freq, Pxx_h_2v_v = calc_Pxx_velmap(h_2v_v)
plot_Pxx_2D(freq, Pxx_h_2v_v, title="PSD h2v intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=d)


k_2r_v = filter_velocity_field(rp[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, 0], dd=dd_param)
plot_velocity_map(k_2r_v, title="k2r velocity @ mu="+mu_angle, aspect=asp, d=d)
bp_new = np.swapaxes(rp[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 1, 0], 
                     0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param),
                   title="k2r intensity @ mu="+mu_angle, aspect=asp, 
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.3,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.5,
                   d=d)
freq, Pxx_k_2r_v = calc_Pxx_velmap(k_2r_v)
plot_Pxx_2D(freq, Pxx_k_2r_v, title="PSD k2r intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=d)


h_2r_v = filter_velocity_field(rp[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, 1], dd=dd_param)
plot_velocity_map(h_2r_v, title="h2r velocity @ mu="+mu_angle, aspect=asp, d=d)
bp_new = np.swapaxes(rp[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 1, 1], 
                     0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param),
                   title="h2r intensity @ mu="+mu_angle, aspect=asp, 
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.3,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.5,
                   d=d)
freq, Pxx_h_2r_v = calc_Pxx_velmap(h_2r_v)
plot_Pxx_2D(freq, Pxx_h_2r_v, title="PSD h2r intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=d)

k_3_v = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                 slit_limits[0]:slit_limits[1], 0, 0], dd=dd_param)
plot_velocity_map(k_3_v, title="k3 velocity @ mu="+mu_angle, aspect=asp, d=d)
bp_new = np.swapaxes(lc[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 1, 0], 
                     0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param), 
                   title="k3 intensity @ mu="+mu_angle, aspect=asp, 
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.3,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.5,
                   d=d)
freq, Pxx_k_3_v = calc_Pxx_velmap(k_3_v)
plot_Pxx_2D(freq, Pxx_k_3_v, title="PSD k3 intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=d)

h_3_v = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                 slit_limits[0]:slit_limits[1], 0, 1], dd=dd_param)
plot_velocity_map(h_3_v, title="h3 velocity @ mu="+mu_angle, aspect=asp, d=d)
bp_new = np.swapaxes(lc[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 1, 1], 
                     0, 1)
plot_intensity_map(average_velmap_slitwise(bp_new, slit_av_param),
                   title="h3 intensity @ mu="+mu_angle, aspect=asp, 
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.3,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.5,
                   d=d)
freq, Pxx_h_3_v = calc_Pxx_velmap(h_3_v)
plot_Pxx_2D(freq, Pxx_h_3_v, title="PSD h3 intensity @ mu="+mu_angle,
            aspect=asp_pxx, d=d)

