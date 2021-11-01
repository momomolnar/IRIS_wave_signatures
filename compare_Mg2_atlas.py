#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:08:43 2021

Compare the Mg II h/k spectrum from RH with the one from the atlas 
(Shaath and Lemaire, 95, ApJ)

@author: molnarad
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as pl
from scipy.io import readsav

pwd = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Atomic_data/"
atlas = "Staath_Lemaire_1995_MgII.csv"
RH = "MgII_hk_FALC.sav"

atlas = np.loadtxt(pwd+atlas, delimiter=",").T

RH = readsav(pwd+RH)
RH_I = RH["I"]
RH_lambda = RH["lambda"]


pl.figure(dpi=250)
pl.plot(atlas[0, :], atlas[1, :], 'r.', label="Atlas\nStaath+Lemaire",
        markersize=0.3)
pl.plot(RH_lambda, RH_I[0, :]*1e11, 'g--', label="RH", alpha=0.5)
pl.xlabel("Wavelength [nm]")
pl.ylabel("Intensity")
pl.ylim(0, 200)
pl.xlim(279, 281)
pl.legend()
pl.show()