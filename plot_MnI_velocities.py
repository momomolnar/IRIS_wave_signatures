#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:06:37 2020

Plot the velocities from the SiIV data 

@author: molnarad
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from scipy import signal
from IRIS_lib import plot_velocity_map, filter_velocity_field, plot_Pxx_2D


d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/CH_data/"
filename = "MnI_2801.9_0.92.sav"
mu_angle = filename[-8:-4]

aa = io.readsav(d+filename)

fit_params = aa["fit_params"]

velocities = fit_params[1, :, :]

rest_wave = np.median(velocities)

velocities = (velocities - rest_wave)*3e5/2801.9

vel_reduced = filter_velocity_field(velocities,  
                                    dt=5, dx=5, dd=1, degree=1)

v_lim = 2

plot_velocity_map(vel_reduced.T, vmin=-1*v_lim, vmax=v_lim, aspect=0.1, 
                  title=" CH Mn I Velocity " + mu_angle)

im1 = pl.imshow(velocities.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
pl.colorbar(im1)
pl.title("CH Mn I Velocity " + mu_angle)
pl.show()


freq, _ = signal.periodogram(vel_reduced[3:, 0], fs=1/16.7)
Pxx     = [signal.periodogram(el, fs=1/16.7)[1] for el in vel_reduced[3:, :].T]
Pxx     = np.array(Pxx)

pl.plot(freq*1e3, np.nanquantile(Pxx[50:-50, :]/1e3, 0.5, axis=0))
#pl.yscale("log")
pl.xlim(1, )
pl.ylim(1e-4, 1e0)
pl.ylabel("PSD [(km/s)$^2$/mHz]")
pl.xlabel("Frequency [mHz]")
pl.yscale("log")
pl.xscale("log")
pl.grid(alpha=0.4)
pl.show()

plot_Pxx_2D(freq, Pxx, aspect=1/0.3)