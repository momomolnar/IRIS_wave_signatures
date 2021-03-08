#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 00:06:44 2021
Try to fit a model to the data for mu angles
@author: molnarad
"""

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as pl

def parabola(x, a, b):
    return a*x*x - b

def calc_params(popt):
    v_perp = np.sqrt(-1 * popt[1])
    v_par  = np.sqrt(popt[0] - popt[1])
    return v_par, v_perp

data = np.array([3208.4,  5300, 4493.5 , 3823.7, 4724.0])*0.000499001
mu = [0.2, 0.4, 0.6, 0.8, 1.0]

popt, pcov = optimize.curve_fit(parabola, mu, data, p0=[-1000, 2000], sigma=0.1*data)

x = np.linspace(0, 1, num=100)

pl.figure(dpi=250)
pl.errorbar(mu, data, yerr=(0.1*data), fmt='o')
pl.plot(x, parabola(x, popt[0], popt[1]))
pl.xlabel("$\\mu$ angle")
pl.ylabel("Power [(km/s)$^2$/mHz]")
pl.show()

v_par, v_per = calc_params(popt)
print(f"The v_par, v_per are: {v_par:.2f}, {v_per:.2f}!")