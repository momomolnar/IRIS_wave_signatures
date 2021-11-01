#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 16:42:44 2021

Plot different PSDs from different 

@author: molnarad
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from scipy import stats as stats

#from PYBIS import construct_name, load_fft_data_simple


data_dir_IBIS = ('/Users/molnarad/CU_Boulder/Work/'
                +'Chromospheric_business/Comps_2/comps_code/')

IBIS_file = "PSD_ROS.npz"

data_dir_IRIS = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/code/"
IRIS_MnI_file  = "IRIS_MnI_Pxx.npz"
IRIS_MgII_file = "IRIS_MgIIh3_Pxx.npz"



IBIS_data = np.load(data_dir_IBIS + IBIS_file)
IBIS_PSD = 10**IBIS_data["fft_mean_data"][-2]
IBIS_freq = IBIS_data["freq"]

IRIS_MnI_data = np.load(data_dir_IRIS + IRIS_MnI_file)
IRIS_freq_MnI = IRIS_MnI_data["arr_1"]
IRIS_MnI_data = IRIS_MnI_data["arr_0"]
IRIS_MnI_data = np.nanquantile(IRIS_MnI_data[50:-70, :], 0.5, axis=0)


IRIS_MgII_data = np.load(data_dir_IRIS + IRIS_MgII_file)
IRIS_freq      = IRIS_MgII_data["arr_1"]
IRIS_MgII_data = IRIS_MgII_data["arr_0"]
IRIS_MgII_data = np.nanquantile(IRIS_MgII_data[50:-70, :], 0.5, axis=0)

pl.figure(dpi=250)
pl.loglog(IBIS_freq, IBIS_PSD, 'r.--', label="IBIS Ca II IR")
pl.loglog(IRIS_freq_MnI*1e3, IRIS_MnI_data/1e3, 'g.--', label="IRIS Mn I")
pl.loglog(IRIS_freq*1e3, IRIS_MgII_data/1e3, 'b.--', label="IRIS Mg II k3")
pl.grid(alpha=0.25)
pl.legend()
pl.xlabel("Frequency [mHz]")
pl.ylabel("Power [(km/s)$^2$/mHz]")