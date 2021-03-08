#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:58:57 2021

Compare IBIS IRIS dataset 

@author: molnarad
"""

import numpy as np
from scipy import io 
from matplotlib import pyplot as pl 
from skimage.transform import downscale_local_mean

with np.load("Power_mg_k3_line_IRIS_IBIS_raster.npz") as aa:
    power_map_IRIS = aa["arr_0"]
    
power_map_IRIS = downscale_local_mean(power_map_IRIS, (12, 2))
    
data_dir = '/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Comps_2/comps_data/'
fft_file_ca = data_dir+'IBIS/fft.nb.vel.lc.155413.ca8542.sav'

a = io.readsav(fft_file_ca, verbose=False)
fft_data = np.flip(a.fft_data, axis=1)
frequency = a.freq1
fft_power = a.fft_powermap

power_IBIS = np.sum(fft_data[11:23, :, :], axis=0) / frequency[1]
power_IBIS = downscale_local_mean(power_IBIS, (12, 2))

pl.hist(np.log10(power_IBIS+1e-15).flatten(), 
        bins=100, range=[-3, 1])
pl.title("IBIS power")
pl.show()

pl.hist(np.log10(power_map_IRIS+1e-15).flatten(), bins=100, range=[-3, -1])
pl.title("IRIS power")
pl.show()


pl.hist2d(np.log10(power_IBIS+1e-15).flatten(),
          np.log10(power_map_IRIS+1e-15).flatten(), density=True, bins=50, 
          range=[[-1.3, .7], [-3.2, -.8]])