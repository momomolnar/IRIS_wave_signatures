#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:12:52 2021

Calculate the Intensity -- Velocity phase in the IRIS data for the k2v line 
feature for both plage and for QS.

Use the provided reduced datasets from .npz archives

@author: molnarad
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from IRIS_lib import plot_velocity_map, filter_velocity_field, remove_nans
from IRIS_lib import angle_mean, calc_phase_diff, plot_intensity_map, average_velmap_slitwise


d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/QS_data/"
filename_Mg = "iris_l2_201311167_QS_SitAndStare_mu=0.8.sav"

title = 'IRIS Mg II k2v Velocity-Intensity'
label = 'Vel-Intensity'
units = 'km s$^{-1}$'
t0 = 0
dt = 16.7 # In seconds

mu_angle = filename_Mg[-7:-4]
#Load velocity sequence 1
aa = io.readsav(d+filename_Mg)
fit_params = aa["bp"]

velocities = fit_params[:, :, 0, 1]
MgII_k2v = filter_velocity_field(velocities,
                                 dt=3, dd=3, degree=2)

plot_velocity_map(MgII_k2v, vmin=-6, vmax=6, aspect=0.3,
                  title="Mg II k2v Doppler velocity ")

#Load velocity sequence 2


# MgII_k3 = filter_velocity_field(velocities,
#                                 dv=7, dd=5, degree=2)

bp_new = np.swapaxes(fit_params[0:-3, :, 1, 0], 0, 1)
intensity = bp_new

intensity = np.array(intensity)
intensity[np.isnan(intensity)] = -200
intensity[np.where(intensity==-200)] = np.median(intensity)
plot_intensity_map(intensity,
                   title="CH k2v intensity @ mu=1.0", 
                   aspect=.3, 
                   vmin=np.amin(bp_new[~(np.isnan(bp_new))])*1.2,
                   vmax=np.amax(bp_new[~(np.isnan(bp_new))])*.5,
                   d=d, cadence = dt)


angles, cor_sig_all, freq, coi_map = calc_phase_diff(MgII_k2v, intensity, 
                                                     use_cache=True)

#compute ALL the phases averaged
angles_average = [angle_mean(el.flatten()) for el in
                  angles]
angles_average = np.array(angles_average)

#Compute only the statistically SIGNIFICANT phases 
angles_masked = angles * (cor_sig_all > 1) 
angles_masked *= coi_map[:, :, None]
phase_freq = np.zeros(angles_masked.shape[0])

for el in range(angles_masked.shape[0]): 
    temp = angles_masked[el, :, 100:-100]
    temp[np.isnan(temp)] = 0
    temp = np.ma.masked_equal(temp, 0.0)
    temp1 = temp.compressed()
    phase_freq[el] = angle_mean(temp1.flatten())

pl.figure(dpi=250)
pl.plot(freq*1e3, angles_average*180/3.1415, label="All angles")
pl.plot(freq*1e3, phase_freq*180/3.1415,
        label="Statistically significant")
pl.xlabel("Frequency [mHz]")
pl.ylim(-180, 180)
pl.grid()
pl.legend()
pl.title(f"Phase diff between Mg II k2v vel and intensity @ mu={mu_angle}")
pl.ylabel("Phase difference [deg]")
