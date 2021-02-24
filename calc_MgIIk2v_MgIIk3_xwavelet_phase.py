#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:55:33 2021

Calculate the phase difference between Mg II k2v and the k3 component
@author: molnarad
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from IRIS_lib import plot_velocity_map, filter_velocity_field, remove_nans
from IRIS_lib import angle_mean, calc_phase_diff


d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/QS_data/"
filename_MnI = "iris_l2_20131116_073345_3803010103_raster_t000_r00000MnI_2801.9.sav"
filename_Mg = "iris_l2_20131116_073345_3803010103_raster_t000_r00000_mg2_vel.sav"

title = 'IRIS Mg II k2v - k3'
label = 'Vel'
units = 'km s$^{-1}$'
t0 = 0
dt = 16.7 # In seconds

#Load velocity sequence 1
aa = io.readsav(d+filename_Mg)
fit_params = aa["bp"]

velocities = fit_params[:, :, 0, 0]
MgII_k2v = filter_velocity_field(velocities,
                                 dv=7, dd=5, degree=2)

plot_velocity_map(MgII_k2v, vmin=-7, vmax=7, aspect=0.3,
                  title="Mg II k2v Doppler velocity ")

#Load velocity sequence 2
fit_params = aa["lc"]

velocities = fit_params[:, :, 0, 0]

MgII_k3 = filter_velocity_field(velocities,
                                dv=7, dd=5, degree=2)

plot_velocity_map(MgII_k3, vmin=-7, vmax=7, aspect=0.3,
                  title="Mg II k3 Doppler velocity ")

angles, cor_sig_all, freq = calc_phase_diff(MgII_k2v, MgII_k3)

#compute ALL the phases averaged
angles_average = [angle_mean(el.flatten()) for el in
                  angles]
angles_average = np.array(angles_average)

#Compute only the statistically SIGNIFICANT phases 
angles_masked = angles*cor_sig_all
phase_freq = np.zeros(70)
for el in range(70): 
    temp = angles_masked[el, :, :-70]
    temp[np.isnan(temp)] = 0
    temp = np.ma.masked_equal(temp, 0.0)
    temp1 = temp.compressed()
    phase_freq[el] = angle_mean(temp1.flatten())

pl.figure(dpi=250)
pl.plot(freq*1e3, angles_average*180/3.1415, label="All angles")
pl.plot(freq[2:]*1e3, phase_freq*180/3.1415,
        label="Statistically significant")
pl.xlabel("Frequency [mHz]")
pl.grid()
pl.legend()
pl.title("Phase diff between Mg II k2v and k3")
pl.ylabel("Phase difference [deg]")
