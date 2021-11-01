#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:09:27 2021

compare the IRIS/IBIS/ALMA data set for 23rd of April 2017

@author: molnarad
"""

import astropy 
import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from IRIS_lib import plot_velocity_map, filter_velocity_field, calc_Pxx_velmap

data_dir = ("/Users/molnarad/CU_Boulder/Work/Chromospheric_business"
            + "/IRIS_waves/IRIS_data/Apr_2017_data/")
IRIS_filename = "IRIS_raster_IBIS_aligned_MnI.2017Apr23.sav"

aa = io.readsav(data_dir+IRIS_filename)
#pl.imshow(aa["data_cube_bp_v"][0, 0, :, :], origin="lower", cmap="gist_gray",
#          vmin=0)

data_regions = [341, 362, 383, 404, 425, 445, 465, 466, 486, 507]

power_map = np.zeros((1000, 1000))

velocities = (aa["data_cube_lc_v"] 
              - np.median(aa["data_cube_lc_v"][:, :, 345:507])) * 3e5 / 2801.5

for el in range(len(data_regions)-1):
    print(f"Calculating slit location {el}")
    xx = int((data_regions[el] + data_regions[el+1]) // 2)
    vel_reduced_k = filter_velocity_field(velocities[:, :, xx],
                                          dt=2, dx=2, dd=3, degree=1)
    mask = np.where(np.log10(np.amax(np.abs(vel_reduced_k[:]), axis=1)) > 1.5)
    vel_reduced_k[mask, :] = 0 
    
    freq, Pxx_k = calc_Pxx_velmap(vel_reduced_k, fsi=1/25)
    
    Pxx_average = Pxx_k
    
    power = np.sum(Pxx_average[:, 37:111], axis=1)*freq[1] # - noise_estimate    

    power_map[1:, data_regions[el]:data_regions[el+1]] = power[:, None] 
    
    # plot_Pxx_2D(freq, ((Pxx_h_2r_v0+Pxx_h_2r_v1)/2)[250:-150, :], title="", 
    #             vimaspect=30, vmina=-1., vmaxa=0.0)

pl.imshow(np.log10(power_map+1e-15), origin="lower", vmin=-1, vmax=0.3)

np.savez("Power_mg_k3_line_IRIS_IBIS_raster_MnI.npz", power_map)