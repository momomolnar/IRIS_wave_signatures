#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:36:15 2021

Plot the mu dependence of the observed velocity fluctuations in the Mg k/h lines

@author: molnarad
""" 
import numpy as np
from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D
from matplotlib import pyplot as pl

file_name = ["../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.2.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.4.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.6.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.8.sav",
             "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=1.0.sav"]

d= "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Meetings/Meetings_with_Steve/2021_02_25/"

t_limits = [1, -2]
slit_limits = [20,-40]
lim1 = 10
lim2 = 30

P_2v = np.zeros(len(file_name))
P_2r = np.zeros(len(file_name))
P_3 = np.zeros(len(file_name))

dt_val = 7
dx_val = 10

for el in range(len(file_name)):
    file = file_name[el]
    print(f"Processing file {file[-10:-4]}")

    cadence = 16.7
    
    bp, rp, lc = load_velocity_mg2(file)

    k_2v_v = filter_velocity_field(bp[t_limits[0]:t_limits[1], 
                                      slit_limits[0]:slit_limits[1], 0, 1], 
                                   dt=dt_val, dx=dx_val, dd=5, degree=1)
    h_2v_v = filter_velocity_field(bp[t_limits[0]:t_limits[1], 
                                      slit_limits[0]:slit_limits[1], 0, 0], 
                                   dt=10, dx=10, dd=5, degree=1)
    av_map_2v = 0.5*(h_2v_v + k_2v_v)
    freq, Pxx_2v_v = calc_Pxx_velmap(av_map_2v, fsi=1/cadence)
    Pxx_2v_noise = np.median(Pxx_2v_v[:, -20:-1], axis=1)
    Pxx_2v_v -= Pxx_2v_noise[:, None]
    Pxx_2v_ave = np.nanquantile(Pxx_2v_v, .50, axis=0)
    P_2v[el] = np.sum(Pxx_2v_ave[lim1:lim2])*freq[1]

    print("    2v component done")

    k_2r_v = filter_velocity_field(rp[t_limits[0]:t_limits[1], 
                                      slit_limits[0]:slit_limits[1], 0, 1], 
                                   dt=10, dx=10, dd=5, degree=1)
    h_2r_v = filter_velocity_field(rp[t_limits[0]:t_limits[1], 
                                      slit_limits[0]:slit_limits[1], 0, 0], 
                                   dt=10, dx=10, dd=5, degree=1)
    av_map_2r = 0.5*(h_2r_v + k_2r_v)
    freq, Pxx_2r_v = calc_Pxx_velmap(av_map_2r, fsi=1/cadence)
    Pxx_2r_noise = np.median(Pxx_2r_v[:, -20:-1], axis=1)
    Pxx_2r_v -= Pxx_2r_noise[:, None]
    Pxx_2r_ave = np.nanquantile(Pxx_2r_v, .50, axis=0)
    P_2r[el] = np.sum(Pxx_2r_ave[lim1:lim2])*freq[1]

    print("    2r component done")

    k_3_v = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                      slit_limits[0]:slit_limits[1], 0, 1], 
                                   dt=10, dx=10, dd=5, degree=1)
    h_3_v = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                      slit_limits[0]:slit_limits[1], 0, 0], 
                                   dt=10, dx=10, dd=5, degree=1)
    av_map_3 = 0.5*(h_3_v + k_3_v)
    freq, Pxx_3_v = calc_Pxx_velmap(av_map_3, fsi=1/cadence)
    Pxx_3_noise = np.median(Pxx_3_v[:, -20:-1], axis=1)
    Pxx_3_v -= Pxx_3_noise[:, None]
    Pxx_3_ave = np.nanquantile(Pxx_3_v, .50, axis=0)
    P_3[el] = np.sum(Pxx_3_ave[lim1:lim2])*freq[1]
    print("    3 component done")

mu_angles = [0.2, 0.4, 0.6, 0.8, 1.0]
pl.figure(figsize=(4,3), dpi=250)
pl.plot(mu_angles, P_2r, '.--', label="2r")
pl.plot(mu_angles, P_2v, '.--', label="2v")
pl.plot(mu_angles, P_3, '.--', label="3")
pl.legend(title="Spectral feature")
pl.ylabel("Velocity rms [(km/s)$^2$]")
pl.xlabel("$\\mu$ angle")
pl.title("Mg II h\&k Doppler velocity fluctuations")
pl.grid(alpha=0.5)