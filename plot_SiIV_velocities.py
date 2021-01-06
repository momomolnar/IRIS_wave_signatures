#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:06:37 2020

Plot the velocities from the SiIV data 

@author: molnarad
"""

import astropy 
import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from IRIS_lib import plot_velocity_map, filter_velocity_field

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/"
filename = "iris_l2_20131116_073345_3803010103_raster_t000_r00000SiIV_vel.sav"

aa = io.readsav(d+filename)

fit_params = aa["fit_params"]

velocities = fit_params[1, :, :]

rest_wave = np.median(velocities)

velocities = (velocities - rest_wave)*3e5/rest_wave

vel_reduced = filter_velocity_field(velocities.T,  
                                    dv=1, dd=1, degree=0)

plot_velocity_map(vel_reduced, vmin=-8, vmax=8, aspect=0.1)