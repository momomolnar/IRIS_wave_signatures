#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 09:22:34 2021

Make a violin plot for the v_rms for the different solar distribution of v_rms

@author: molnarad
"""

# library & dataset
import seaborn as sns
import numpy as np
from scipy import io
import matplotlib.pyplot as pl
from scipy import signal
from astropy.io import fits
from IRIS_lib import plot_velocity_map, filter_velocity_field, plot_Pxx_2D
from IRIS_lib import filter_velocity_field, load_velocity_mg2
from IRIS_lib import plot_intensity_map
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from scipy import stats as stats

from PYBIS import construct_name, load_fft_data_simple

#Load some IBIS data

fontsize_legend = 7
pl.style.use('science')

def count_zeroes_in_array(array):
    counter = 0
    for el in array.ravel():
        if el == False:
            counter += 1

    return counter

dir_IBIS = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Comps_2/comps_code"

labels = ['plage', 'internetwork']
#labels = ['penumbra']
#labels = ['network']
fft_mean_data = np.zeros((5, 61))

masks_file = 'Masks_FOV2.npz'
i = 0


data_dir = ('/Users/molnarad/CU_Boulder/Work/'
                +'Chromospheric_business/Comps_2/comps_data/')
spec_property = 'vel.lc'
time_stamp    = 170444
freq, fft_data = load_fft_data_simple(data_dir, spec_property, time_stamp)



fft_data = (fft_data)
dx = 80
ddfreq = freq[1] - freq[0]

freq_int = [5, 20]
freq_lim = (freq_int // freq[1]) + 1 

d = {"Region":[], "RMS":[], "Spectral Line":[]}
IBIS_data_frame = pd.DataFrame(d)

for mask_el in labels:
    print(f"{mask_el} is being processed.")
    with np.load(masks_file) as file:
        mask = np.array(file[mask_el])
    if mask_el == "internetwork":
        mask_name = "QS"
    if mask_el == "plage":
        mask_name = "Plage"
    for ii in range(125, 225):
        for jj in range(25, 200):
            if mask[ii*4, jj*4]!=0:
                rms_vel = np.sum(fft_data[int(freq_lim[0]):int(freq_lim[1]),
                                          ii*4, jj*4])*freq[1]
                if (rms_vel > 0) and (np.isnan(rms_vel) != True):
                    d = {"Region":mask_name, "RMS":np.log10(rms_vel), "Spectral Line":"CaII IR"}
                    IBIS_data_frame = IBIS_data_frame.append(d, ignore_index=True)

                #print("Successfully attached to dataframe")

# load IRIS QS Mn I data

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/QS_data/"
filename = ["MnI_2801.9_mu=1.0.sav"]
cadence = [5]
num_files = len(filename)
mu_angles = np.zeros(num_files)
v_rms = np.zeros(num_files)

freq_int = [5e-3, 20e-3]

Pxx_noise_ave = np.zeros(num_files)

for el1 in range(num_files):
    aa = io.readsav(d+filename[el1])
    
    mu_angle = filename[el1][-7:-4]
    mu_angles[el1] = mu_angle

    fit_params = aa["fit_params"]
    velocities = fit_params[1, :, :]
    rest_wave = np.median(velocities)
    velocities = (velocities - rest_wave)*3e5/2801.9
    vel_reduced = filter_velocity_field(velocities,  
                                        dt=5, dx=5, dd=1, degree=1)
    v_lim = 2

    plot_velocity_map(vel_reduced.T, vmin=-1*v_lim, vmax=v_lim, aspect=0.1, 
                      title="CH Mn I Velocity " + mu_angle)

    #im1 = pl.imshow(vel_reduced.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
    #pl.colorbar(im1)
    #pl.title("CH Mn I Velocity " + mu_angle)
    #pl.show() 
    
    lim_vel = [50, -50]
    
    
    freq, _ = signal.periodogram(vel_reduced[:, 0], fs=1/cadence[el1])
    Pxx     = [signal.periodogram(el, fs=1/cadence[el1])[1] for el in 
               vel_reduced[:, lim_vel[0]:lim_vel[1]].T]
    Pxx     = np.array(Pxx)
       
    freq_lim = (freq_int // freq[1])
    Pxx_noise = np.nanquantile(Pxx[:, -15:], 0.5, axis=1)
    Pxx_denoise = Pxx - Pxx_noise[:, None]
    # Add noise estimate
    
    for el2 in range(Pxx[50:-50, :].shape[0]):
        rms_vel = np.sum(Pxx_denoise[el2+50, int(freq_lim[0]):int(freq_lim[1])]) * freq[1]
        #print(rms_vel)
        if (rms_vel > 0) and (np.isnan(rms_vel) != True):
            d =   {"Region":"QS", "RMS":np.log10(rms_vel), "Spectral Line":"MnI 280.19 nm"}
            IBIS_data_frame = IBIS_data_frame.append(d, ignore_index=True )
    #print(f"V_RMS_{el1} is {v_rms[el1]} $\pm$ {Pxx_noise_ave[el1]}.") 
    
d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/QS_data/"
filename = ["iris_l2_201311167_QS_SitAndStare_mu=1.0.sav"]

cadence = [16.7]
#cadence = [17, 17, 4, 5, 9, 9, 5, 9] # without the mu=0.44 dataset 
num_files = len(filename)
mu_angles = np.zeros(num_files)
v_rms = np.zeros(num_files)
v_rms_noise = np.zeros(num_files)
t_limits = [1, -2]
line_index = 0
slit_limits = [10, -10]

freq_int = [5e-3, 15e-3]

Pxx_noise_ave = np.zeros(num_files)

lim_vel = [1, -1]

for el1 in range(num_files):
    #plot intensity 
    #pick only top 50% of intensity 
    # compare intensity fluctuations with velocity fluctuations -- if it's alfvenic waves, no correlation!
    mu_angle = filename[el1][-7:-4]

    mu_angles[el1] = mu_angle
    bp, rp, lc = load_velocity_mg2(d + filename[el1])
    vel_reduced = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, 1], 
                               dt=7, dx=7, dd=3, degree=1)
    v_lim = 7

    plot_velocity_map(vel_reduced, vmin=-1*v_lim, vmax=v_lim, aspect=0.1, 
                      title="Plage Mg II Velocity " + mu_angle)
    bp_new = np.swapaxes(lc[t_limits[0]:t_limits[1], slit_limits[0]:slit_limits[1], 1, line_index], 0, 1)
    plot_intensity_map(bp_new,
                   title="Plage k2v intensity @ mu="+mu_angle, 
                   aspect=0.1, 
                   vmin=150, #np.amin(bp_new[~(np.isnan(bp_new))])*1.2,
                   vmax=1000, #np.amax(bp_new[~(np.isnan(bp_new))])*0.8,
                   d=d, cadence = cadence[el1])

    #im1 = pl.imshow(vel_reduced.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
    #pl.colorbar(im1)
    #pl.title("CH Mn I Velocity " + mu_angle)
    #pl.show() 
    

    
    
    freq, _ = signal.periodogram(vel_reduced[0, :], fs=1/cadence[el1])
    Pxx     = [signal.periodogram(el, fs=1/cadence[el1])[1] for el in 
               vel_reduced[lim_vel[0]:lim_vel[1], :]]
    Pxx     = np.array(Pxx)
    
    plot_Pxx_2D(freq, Pxx, aspect=1/0.3/0.3)
    
    freq_lim = (freq_int // freq[1])
    
    Pxx_noise = np.nanquantile(Pxx[:, -15:], 0.5, axis=1)
    Pxx_denoise = Pxx - Pxx_noise[:, None]
    # Add noise estimate
    for el2 in range(Pxx_denoise[50:-50, :].shape[0]):
        rms_vel = np.sum(Pxx_denoise[50+el2, int(freq_lim[0]):int(freq_lim[1])]) * freq[1]
        #print(rms_vel)
        if (rms_vel > 0) and (np.isnan(rms_vel) != True):
            d =   {"Region":"QS", "RMS":np.log10(rms_vel), "Spectral Line":"MgII k"}
            IBIS_data_frame = IBIS_data_frame.append(d, ignore_index=True)
            
    print(f"V_RMS_{el1} is {v_rms[el1]} $\pm$ {Pxx_noise_ave[el1]}.") 


# load IRIS Plage data

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Plage_data/"
filename = ["MnI_2801.9_1.00.sav"]
cadence = [16.7]
num_files = len(filename)
mu_angles = np.zeros(num_files)
v_rms = np.zeros(num_files)

freq_int = [5e-3, 20e-3]

Pxx_noise_ave = np.zeros(num_files)

for el1 in range(num_files):
    aa = io.readsav(d+filename[el1])
    
    mu_angle = filename[el1][-7:-4]
    mu_angles[el1] = mu_angle

    fit_params = aa["fit_params"]
    velocities = fit_params[1, :, :]
    rest_wave = np.median(velocities)
    velocities = (velocities - rest_wave)*3e5/2801.9
    vel_reduced = filter_velocity_field(velocities,  
                                        dt=5, dx=5, dd=1, degree=1)
    v_lim = 2

    plot_velocity_map(vel_reduced.T, vmin=-1*v_lim, vmax=v_lim, aspect=0.1, 
                      title="Plage Mn I Velocity " + mu_angle)

    #im1 = pl.imshow(vel_reduced.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
    #pl.colorbar(im1)
    #pl.title("CH Mn I Velocity " + mu_angle)
    #pl.show() 
    
    lim_vel = [50, -50]
    
    
    freq, _ = signal.periodogram(vel_reduced[:, 0], fs=1/cadence[el1])
    Pxx     = [signal.periodogram(el, fs=1/cadence[el1])[1] for el in 
               vel_reduced[:, lim_vel[0]:lim_vel[1]].T]
    Pxx     = np.array(Pxx)
       
    freq_lim = (freq_int // freq[1])
    Pxx_noise = np.nanquantile(Pxx[-100:, :], 0.5, axis=0)
    Pxx_denoise = Pxx - Pxx_noise[None, :]
    # Add noise estimate
    
    for el2 in range(Pxx[100:700, :].shape[0]):
        #rms_vel = np.var(vel_reduced[:, el2])
        rms_vel = np.sum(Pxx_denoise[el2+100, int(freq_lim[0]):int(freq_lim[1])]) * freq[1]
        #print(rms_vel)
        if (rms_vel > 0) and (np.isnan(rms_vel) != True):
            d =   {"Region":"Plage", "RMS":np.log10(rms_vel), "Spectral Line":"MnI 280.19 nm"}
            IBIS_data_frame = IBIS_data_frame.append(d, ignore_index=True)
    #print(f"V_RMS_{el1} is {v_rms[el1]} $\pm$ {Pxx_noise_ave[el1]}.") 
    

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Plage_data/"
filename = ["iris_l2_20140918_080253_3820257453_raster_t000_r00000__1.00_mg2_vel.sav"]

cadence = [5]
#cadence = [17, 17, 4, 5, 9, 9, 5, 9] # without the mu=0.44 dataset 
num_files = len(filename)
mu_angles = np.zeros(num_files)
v_rms = np.zeros(num_files)
v_rms_noise = np.zeros(num_files)
t_limits = [1, -2]
line_index = 0
slit_limits = [10, -10]

freq_int = [5e-3, 15e-3]

Pxx_noise_ave = np.zeros(num_files)

lim_vel = [1, -1]

for el1 in range(num_files):
    #plot intensity 
    #pick only top 50% of intensity 
    # compare intensity fluctuations with velocity fluctuations -- if it's alfvenic waves, no correlation!
    mu_angle = "1.00"

    mu_angles[el1] = mu_angle
    bp, rp, lc = load_velocity_mg2(d + filename[el1])
    vel_reduced = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, 1],  
                                        dt=5, dx=5, dd=1, degree=1)
    v_lim = 2


    plot_velocity_map(vel_reduced, vmin=-1*v_lim, vmax=v_lim, aspect=0.1)
    #im1 = pl.imshow(vel_reduced.T, cmap="bwr", vmin=-1*v_lim, vmax=v_lim)
    #pl.colorbar(im1)
    #pl.title("CH Mn I Velocity " + mu_angle)
    #pl.show() 
    

    
    
    freq, _ = signal.periodogram(vel_reduced[0, :], fs=1/cadence[el1])
    Pxx     = [signal.periodogram(el, fs=1/cadence[el1])[1] for el in 
               vel_reduced[lim_vel[0]:lim_vel[1], :]]
    Pxx     = np.array(Pxx)
    
    plot_Pxx_2D(freq, Pxx, aspect=1/0.3/0.3)
    
    freq_lim = (freq_int // freq[1])
    
    Pxx_noise = np.nanquantile(Pxx[:, -15:], 0.5, axis=1)
    Pxx_denoise = Pxx - Pxx_noise[:, None]
    # Add noise estimate
    for el2 in range(Pxx_denoise[50:-50, :].shape[0]):
        rms_vel = np.sum(Pxx_denoise[50+el2, int(freq_lim[0]):int(freq_lim[1])]) * freq[1]
        #print(rms_vel)
        if (rms_vel > 0) and (np.isnan(rms_vel) != True):
            d =   {"Region":"Plage", "RMS":np.log10(rms_vel), "Spectral Line":"MgII k"}
            IBIS_data_frame = IBIS_data_frame.append(d, ignore_index=True)
            
    print(f"V_RMS_{el1} is {v_rms[el1]} $\pm$ {Pxx_noise_ave[el1]}.") 


# plot
g = sns.violinplot(x=IBIS_data_frame["Region"], y=IBIS_data_frame["RMS"], 
                   hue=IBIS_data_frame["Spectral Line"],
                   hue_order=["MnI 280.19 nm", "CaII IR", "MgII k"])
g.set(ylim=(0, 2))
