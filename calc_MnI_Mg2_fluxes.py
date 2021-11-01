#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 01:09:27 2021

Calculate some acoustic fluxes

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
IRIS_MnI_data = np.quantile(IRIS_MnI_data[50:-70, : ], 0.5, axis=0)


IRIS_MgII_data = np.load(data_dir_IRIS + IRIS_MgII_file)
IRIS_freq      = IRIS_MgII_data["arr_1"]
IRIS_MgII_data = IRIS_MgII_data["arr_0"]
IRIS_MgII_data = np.quantile(IRIS_MgII_data[50:-70, :], 0.5, axis=0)

rho_MnI = 10**(-6.75)
cs= 7000
rho_Mg2 = 10**(-10.5)

flux_MnI = np.sum(IRIS_MnI_data[50:-70, 10:30], axis=1) * rho_MnI *  cs
flux_MgII = np.sum(IRIS_MgII_data[50:-70, 10:30], axis=1) * rho_Mg2 *  cs
pl.hist(flux_MnI, label="Mn I")
pl.hist(flux_Mg2, label="Mg 2")
pl.legend()