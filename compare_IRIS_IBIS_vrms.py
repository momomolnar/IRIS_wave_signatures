#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 23:07:00 2021

Compare the IRIS and IBIS V_rms for the different 


@author: molnarad
"""

import numpy as np
from matplotlib import pyplot as pl
from scipy.io import readsav
from scipy import signal
from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D

def prep_data(data_cube):
    """
    Detrend the data in the IRIS datastreams

    Parameters
    ----------
    data_cube : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """


ca_data_dir = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Comps_2/comps_data/IBIS/"
ca_file     = "fft.nb.vel.lc.23Apr2017.target2.all.8542.ser_170444.sav"
a = readsav(ca_data_dir+ca_file, verbose=False)
fft_data = np.flip(a.fft_data, axis=1)
frequency = a.freq1
fft_power = a.fft_powermap
ca_coeff = 1233.5e6

fft_data_IBIS = np.sum(fft_data[2:18, :, :], axis=0)
fft_data_IBIS *= ca_coeff

data_dir = ("/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves"
            + "/IRIS_data/Apr_2017_data/")

file_name ="IRIS_raster_IBIS_aligned.2017Apr23.sav"

sav_data = readsav(data_dir + file_name)
data_cube_bp_v = sav_data["data_cube_bp_v"]
data_cube_bp_v = np.nan_to_num(data_cube_bp_v)

f, Pxx = signal.periodogram(data_cube_bp_v[:, 0, :, :], 
                            fs=1/23, axis=0)

v_k = np.var(data_cube_bp_v[:, 0, :, :], axis=0)
v_h = np.var(data_cube_bp_v[:, 1, :, :], axis=0)

pl.fig(dpi=250)
pl.hist2d(np.log10(np.sum(Pxx[35:70, :, :], axis=0).flatten()+1e-10), 
          np.log10(fft_data_IBIS.flatten())-6, 
          bins=50, range=[[-0, 2.5], [-2, 2]] )
