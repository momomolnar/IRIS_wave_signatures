#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 15:16:22 2021

@author: molnarad
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from scipy import signal
from IRIS_lib import plot_velocity_map, filter_velocity_field, plot_Pxx_2D

def disp_rel(kx, ky, kz, cs=7):
    return cs * np.sqrt(kx**2 + ky**2 + kz**2)

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/CH_data/"
filename = "MnI_2801.9_0.92.sav"
mu_angle = filename[-8:-4]

aa = io.readsav(d+filename)

fit_params = aa["fit_params"]

velocities = fit_params[1, :, :]

rest_wave = np.median(velocities)

velocities = (velocities - rest_wave)*3e5/2801.9

vel_reduced = filter_velocity_field(velocities,  
                                    dt=5, dx=5, dd=1, degree=1)

v_lim = 2

plot_velocity_map(vel_reduced.T, vmin=-1*v_lim, vmax=v_lim, aspect=0.1, 
                  title=" CH Mn I Velocity " + mu_angle)


fft_vel = np.fft.fft2(vel_reduced[2:-3, 10:-30])
P_vel = np.abs(fft_vel)**2

omega_range1 = 0
omega_range2 = 70

kx_range1 = 0
kx_range2 = 120

kx_max = 1 / (0.167 * 700)
omega_max = 1 / (16)


d_omega = omega_max / P_vel.shape[0]
d_kx = kx_max / P_vel.shape[1]

im1 = pl.imshow(np.log10(P_vel[omega_range1:omega_range2, 
                               kx_range1:kx_range2]),
                origin="lower", 
                extent=[0, d_kx*kx_range2*1e3, 0, d_omega*omega_range2*1e3], 
                aspect="auto", vmin=4, vmax=7.5)

pl.ylabel("Frequency $\omega$ [mHz]")
pl.xlabel("k$_x$ [Mm$^{-1}$]")
pl.colorbar(im1)

kx = np.linspace(0, 3, num=10)
pl.plot(kx, disp_rel(kx, 0, 0, cs=7e-3)*1e3, 'r--')
pl.show()
# P_vel dim 0 -- omega  
#   dim 1 -- k_x

