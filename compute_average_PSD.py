#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:48:58 2020

Calculate the average power spectrum of IRIS velocity observation

@author: molnarad
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as pl
from scipy.io import readsav
import scipy as sp

from RHlib import calc_v_lc
from IRIS_lib import load_velocity_mg2, correct_IRIS_noisy_series


def calc_Pxx(file_name):
    
    mu_angle = file_name[-7:-4]
    
    bp, rp, lc = load_velocity_mg2(file_name)

    bp = lc

    bp_v_k_filtered = np.array([correct_IRIS_noisy_series(el, 5) for el in 
                           np.swapaxes(bp[:, 30:-50, 0, 0], 0, 1)])

    bp_I_k_filtered = np.array([correct_IRIS_noisy_series(el, 500) for el in 
                            np.swapaxes(bp[:, 30:-50, 1, 0], 0, 1)])


    pl.figure(dpi=250)
    im1 = pl.imshow(bp[:, 30:-50, 0, 0]- np.mean(np.ma.masked_invalid(bp[:, 100:-100, 0, 0])), 
                    cmap="bwr", vmin=-10, vmax=10, aspect=.25, extent=[0, 768,
                                                                       0, 126*17])
    pl.colorbar(im1, label="Doppler Velocity [km/s]", shrink=.8)
    pl.xlabel("Pixels across the slit")
    pl.title("Mg II k3 line mu="+mu_angle)
    pl.ylabel("Time [s]")
    pl.show()

    pl.figure(dpi=250)
    im1 = pl.imshow(bp_v_k_filtered.T- np.mean(np.ma.masked_invalid(bp_v_k_filtered)), 
                    cmap="bwr", vmin=-10, vmax=10, aspect=.25, extent=[0, 768,
                                                                       0, 126*17])
    pl.colorbar(im1, label="Doppler Velocity [km/s]", shrink=.8)
    pl.title("Mg II k3 line filtered mu="+mu_angle)
    pl.xlabel("Pixels across the slit")
    pl.ylabel("Time [s]")
    pl.show()

    #Use for masking and calculating the average power laws 

    pl.figure(dpi=250)
    im2 = pl.imshow(bp[:, 30:-50, 1, 0],  vmax=800, vmin=0,
                cmap="plasma", aspect=.25, extent=[0, 768, 0, 126*17])
    pl.colorbar(im2, label="Intensity", shrink=.8,)
    pl.xlabel("Pixels across the slit")
    pl.title("Mg II k3 line mu="+mu_angle)
    pl.ylabel("Time [s]")
    pl.show()

    pl.figure(dpi=250)
    im2 = pl.imshow(bp_I_k_filtered.T,  vmax=800, vmin=0,
                    cmap="plasma", aspect=.25, extent=[0, 768, 0, 126*17])
    pl.colorbar(im2, label="Intensity", shrink=.8,)
    pl.xlabel("Pixels across the slit")
    pl.title("Mg II k3 line mu="+mu_angle)
    pl.ylabel("Time [s]")
    pl.show()

    freq_test, Pxx_test = sp.signal.periodogram(bp_v_k_filtered[100, 3:-3], 
                                                fs=1/17)


    P_v_container = np.zeros((bp_v_k_filtered.shape[0], Pxx_test.shape[0])) 
    P_I_container = np.zeros((bp_v_k_filtered.shape[0], Pxx_test.shape[0]))

    for el in range(P_v_container.shape[0]):
        freq, P_v_container[el, :] = sp.signal.periodogram((bp_v_k_filtered[el, 3:-3] 
                                                            - np.mean(bp_v_k_filtered[el, 3:-3])), 
                                                           fs=1/17)
        freq, P_I_container[el, :] = sp.signal.periodogram((bp_I_k_filtered[el, 3:-3] 
                                                            - np.mean(bp_v_k_filtered[el, 3:-3])), 
                                                           fs=1/17)
    
    P_v_containerm = np.array([np.mean(np.ma.masked_invalid(el).compressed()) 
                            for el in np.swapaxes(P_v_container, 0, 1)])
    
    return freq, P_v_container, P_v_containerm
    return freq, P_v_container, P_v_containerm

file_name_mu02 = "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.2.sav"
file_name_mu04 = "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.4.sav"
file_name_mu06 = "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.6.sav"
file_name_mu08 = "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.8.sav"
file_name_mu10 = "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=1.0.sav"

freq, P_v_mu02, P_v_mu02_a = calc_Pxx(file_name_mu02)
freq, P_v_mu04, P_v_mu04_a = calc_Pxx(file_name_mu04)
freq, P_v_mu08, P_v_mu08_a = calc_Pxx(file_name_mu08)
freq, P_v_mu10, P_v_mu10_a = calc_Pxx(file_name_mu10)

pl.figure(dpi=250)
pl.loglog(freq, np.mean(P_v_mu02[390:410, :], axis=0)/1e3,
          "r.--", label="mu=0.2")
pl.loglog(freq, np.mean(P_v_mu04[390:410, :], axis=0)/1e3, "b.--", 
          label="mu=0.4")
pl.loglog(freq, np.mean(P_v_mu08[400:410, :], axis=0)/1e3, "g.--", 
          label="mu=0.8")
pl.loglog(freq, np.mean(P_v_mu10[390:410, :], axis=0)/1e3, "k.--", 
          label="mu=1.0")
pl.ylabel("Power [km$^2$/s$^2$/mHz]")
pl.xlabel("Frequency [Hz]")
pl.title("Average power spectrum of Mg k3 Doppler Vel")
pl.ylim(5e-3, 3e1)
pl.grid(alpha=0.25)
pl.legend()
pl.show()


pl.figure(dpi=250)
im1 = pl.imshow(np.log10(P_v_mu10/1e3), aspect=.05,  
                vmin=-2., vmax=1, extent=[0, freq[-1]*1e3, 0, 646])
pl.colorbar(im1, label="Log10[Power [km^2/s^2/mHz]]")
pl.title("Mu=1.0 PSD 2D")
pl.xlabel("Frequency [mHz]")
pl.ylabel("Slit position [pixel]")
pl.show()

pl.figure(dpi=250)
im1 = pl.imshow(np.log10(P_v_mu08/1e3), aspect=.05,
                vmin=-2., vmax=1, extent=[0, freq[-1]*1e3, 0, 646])
pl.colorbar(im1, label="Log10[Power [km^2/s^2/mHz]]")
pl.title("Mu=0.8 PSD 2D")
pl.xlabel("Frequency [mHz]")
pl.ylabel("Slit position [pixel]")
pl.show()

pl.figure(dpi=250)
im1 = pl.imshow(np.log10(P_v_mu04/1e3), aspect=.05,
                vmin=-2., vmax=1, extent=[0, freq[-1]*1e3, 0, 646])
pl.colorbar(im1, label="Log10[Power [km^2/s^2/mHz]]")
pl.title("Mu=0.4 PSD 2D")
pl.xlabel("Frequency [mHz]")
pl.ylabel("Slit position [pixel]")
pl.show()

pl.figure(dpi=250)
im1 = pl.imshow(np.log10(P_v_mu02/1e3), aspect=.05,  
                vmin=-2., vmax=1, extent=[0, freq[-1]*1e3, 0, 646])
pl.colorbar(im1, label="Log10[Power [km^2/s^2/mHz]]")
pl.title("Mu=0.2 PSD 2D")
pl.xlabel("Frequency [mHz]")
pl.ylabel("Slit position [pixel]")
pl.show()