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
IRIS_filename = "IRIS_raster_IBIS_aligned.2017Apr23.sav"


aa = io.readsav(data_dir+IRIS_filename)
#pl.imshow(aa["data_cube_bp_v"][0, 0, :, :], origin="lower", cmap="gist_gray",
#          vmin=0)


data_regions = [341, 362, 383, 404, 425, 445, 465, 466, 486, 507]

power_map = np.zeros((1000, 1000))

for el in range(len(data_regions)-1):
    print(f"Calculating slit location {el}")
    xx = int((data_regions[el] + data_regions[el+1]) // 2)
    vel_reduced_h = filter_velocity_field(aa["data_cube_lc_v"][:, 0, :, xx],
                                          dv=16, dd=4, degree=2)
    vel_reduced_k = filter_velocity_field(aa["data_cube_lc_v"][:, 1, :, xx],
                                          dv=7, dd=4, degree=2)
    
    freq, Pxx_k = calc_Pxx_velmap(vel_reduced_k, fsi=1/36)
    freq, Pxx_h = calc_Pxx_velmap(vel_reduced_h, fsi=1/36)
    
    Pxx_average = .5*(Pxx_k + Pxx_h)
    
    power = np.sum(Pxx_average[:, 25:51], axis=1)*freq[1] # - noise_estimate    

    power_map[1:, data_regions[el]:data_regions[el+1]] = power[:, None] 
    
    # plot_Pxx_2D(freq, ((Pxx_h_2r_v0+Pxx_h_2r_v1)/2)[250:-150, :], title="", 
    #             vimaspect=30, vmina=-1., vmaxa=0.0)

pl.imshow(np.log10(power_map+1e-15), origin="lower", vmin=-3, vmax=-1)

np.savez("Power_mg_k3_line_IRIS_IBIS_raster.npz", power_map)