#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:29:41 2021


plot Mn I velocity field vs mu 
    
@author: molnarad
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from scipy import signal
from IRIS_lib import plot_velocity_map, filter_velocity_field, plot_Pxx_2D

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/QS_data/"
filename = ["MnI_2801.9_mu=0.2.sav",
            "MnI_2801.9_mu=0.4.sav",
            "MnI_2801.9_mu=0.6.sav",
            "MnI_2801.9_mu=0.8.sav",
            "MnI_2801.9_mu=1.0.sav"]

P = np.zeros(len(filename))
P_noise_ave = np.zeros(len(filename))
mu = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
mu2 = mu * mu

for el in range(len(filename)):
    aa = io.readsav(d+filename[el])

    fit_params = aa["fit_params"]

    velocities = fit_params[1, :, :]

    rest_wave = np.median(velocities)

    velocities = (velocities - rest_wave)*3e5/2801.5

    vel_reduced = filter_velocity_field(velocities,  
                                    dt=1.5, dx=1.5, dd=3, degree=1)

    v_lim = 2.

    plot_velocity_map(vel_reduced.T, vmin=-1*v_lim, vmax=v_lim, aspect=0.3, 
                  title="Mn I 280.15 nm Doppler velocity mu= " + str(mu[el]))

    freq, _ = signal.periodogram(vel_reduced[3:, 0], fs=1/16.7)
    Pxx     = [signal.periodogram(el, fs=1/16.7)[1] for el 
               in vel_reduced[3:, 20:-50].T]
    Pxx     = np.array(Pxx)
    Pxx_ave = np.nanquantile(Pxx, 0.5, axis=0)
    P_noise_ave[el] = np.median(Pxx_ave[-25:-1])
    Pxx_ave -= P_noise_ave[el]
    P[el]   = np.sum(Pxx_ave[10:30])*freq[1]
    
mu_x = np.linspace(0, 1.2, num=100)
coeffs = np.polyfit(mu2, P, deg=1, w=P_noise_ave)
parabola = coeffs[0] * mu_x**2 + coeffs[1]
pl.figure(figsize=(4, 3), dpi=250)
pl.errorbar(mu, P, yerr=P_noise_ave*freq[1]*13, fmt='bo')
pl.plot(mu_x, parabola, 'g--')
pl.ylabel("Velocity fluctuations [(km/s)$^2$]")
pl.xlabel("$\\mu$ angle")
pl.title("Mn I 280.9 nm Doppler velocity fluctuations")
pl.xlim(0.15, 1.021)
pl.ylim(0.045, 0.15)
pl.show()