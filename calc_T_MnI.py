
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the T coefficient for MnI 280.1 nm for all RADYN models

Author: molnarad
2:35PM 08-18-2020
"""

import numpy as np
import scipy as sp
from scipy import io
import matplotlib.pyplot as plt
# from power_uncertain import power_uncertain
import radynpy as rp
from scipy.optimize import curve_fit
import h5py as h5
from scipy import signal

from RHlib import Spectral_Line, Instrument_Profile
from RHlib import RADYN_atmos, calc_v_lc, calc_v_cog

def gauss(x, *p):
    A, mu, sigma, background = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + background

simulation_name = "output_ray_MnI_RADYN.hdf5"

data_dir = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Synthesis/"

atmos = h5.File(data_dir + simulation_name, "r")

I     = atmos["intensity"]
waves = atmos["wavelength"]
tau1_height_a = atmos["tau_one_height"]

waveIndex_1 = 606
waveIndex_2 = 689
wave_lc     = 652

tau1_height = tau1_height_a[:, 0, wave_lc]


dwave = 0.003

wave1 = waves[waveIndex_1]
wave2 = waves[waveIndex_2]

wave_interp = np.arange(wave1, wave2, step=dwave)

num_Tsteps = I.shape[0]
N_skip_tsteps = 100
velocity_l   = np.zeros(num_Tsteps-N_skip_tsteps)
for t in range(N_skip_tsteps, num_Tsteps):
    
    I_interp = np.interp(wave_interp, waves[waveIndex_1:waveIndex_2], 
                         I[t, 0, waveIndex_1:waveIndex_2])
    background_estimate = I_interp[0] 
    I_min = np.amin(I_interp) - background_estimate
    lambda_min = np.mean(wave_interp)
    width = 0.002
    p = [I_min, lambda_min, width, background_estimate]
    coeff, var_matrix = curve_fit(gauss, wave_interp, I_interp, p0=p)
    velocity_l[t-N_skip_tsteps] = coeff[1]
    
vel_median  = np.nanquantile(velocity_l, 0.5)
velocity = 3e5 * (velocity_l - vel_median) / vel_median
    
simulation_name = ["radyn_ulmschneider_100",
                   "radyn_ulmschneider_250",
                   "radyn_ulmschneider_3000"]

data_dir = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Comps_2/comps_data/RADYN_models/"

start_index_list = [1000, 2400, 3000, 4000, 2000]
end_index_list   = [-10, -1000, -1000, -10, -10]

for el in range(2, 3):

    rad = RADYN_atmos(data_dir + simulation_name[el] + '.cdf')

plt.figure(dpi=250)
plt.plot(rad.vz[N_skip_tsteps:, 146]/1e5, 'b.--', label="RADYN@tau=1")
plt.plot(velocity*-1, 'r.--', label="RH MnI")
plt.legend()
plt.xlabel("Time [timestep = 3 s]")
plt.ylabel("Velocity [km/s]")
plt.grid(alpha=0.5)
plt.show()

cadence=3.0 # seconds

MnI_freq_o, MnI_PSD_o = signal.periodogram(velocity, fs=1/cadence)
MnI_freq_m, MnI_PSD_m = signal.periodogram(rad.vz[:, 146], fs=1/cadence)


MnI_freq_o, MnI_PSD_o = signal.periodogram(velocity, fs=1/cadence)
MnI_freq_m, MnI_PSD_m = signal.periodogram(rad.vz[:, 146]/1e5, fs=1/cadence)

plt.figure(dpi=250)
plt.title("PSD for NaD$_1$ line from RADYN_3000 model")
plt.plot(MnI_freq_o, MnI_PSD_o, label="RH MnI")
plt.plot(MnI_freq_m, MnI_PSD_m, label="RADYN")
plt.yscale("log")
plt.legend()
plt.ylim(1e-6, 1e2)
plt.xscale("log")
plt.xlim(1e-3, 50e-3)
plt.grid(alpha=0.5)
plt.show()

max_freq = 60
dfreq    = 5
num_freqs = max_freq // dfreq + 1

MnI_v_PSD_o_mean = [np.mean(MnI_PSD_o[i*19:(19*(i+ 1))]) for i
                     in range(num_freqs)]
MnI_v_PSD_m_mean = [np.mean(MnI_PSD_m[i*19:(19*( i+ 1))]) for i
                     in range(num_freqs)]
freq_range = np.linspace(0, (num_freqs-1)*dfreq,
                         num=num_freqs) + dfreq/2
T_coeff = np.sqrt(np.array(MnI_v_PSD_o_mean)
                  / np.array(MnI_v_PSD_m_mean))

plt.figure(dpi=250)
plt.plot(freq_range, T_coeff)
plt.xlim(0, 30)
plt.title("T coefficient for RADYN 3000 model")
plt.xlabel("Frequency")
plt.ylabel("T coefficient")
plt.grid(alpha=0.5)
plt.ylim(0, 1)

np.savez("T_coeff_MnI.npz", freq_range=freq_range, T_coeff=T_coeff)