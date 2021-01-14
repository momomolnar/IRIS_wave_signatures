#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 18:39:21 2020

Functions to do a few things:
    1) open IRIS data files and load the proper spectral/SJ image;
    2) and calculate the power spectrum of the properties and plot them;
    3) Plot the results from 1) and 2);

@author: molnarad
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as pl
from scipy.io import readsav
import scipy as sp

from RHlib import calc_v_lc

pl.style.use('science')

def remove_nans(array):
    """
    Remove Nans from an array

    Parameters
    ----------
    array : ndarray
        Input array;

    Returns
    -------
    array : ndarray
        Filtered array;

    """
    array1 = array.copy()
    median_array = np.mean(np.ma.masked_invalid(array))
    print(median_array)
    for ii in range(array.shape[0]):
        for jj in range(array.shape[1]):
            if np.isnan(array[ii, jj]) == True:
                array1[ii, jj] = median_array
    
    return array1

def filter_velocity_field(vel_field, dv=8, dd=4, degree=2):
    """
    Filter raw velocity map from IDL input
    by doing the following three steps:
        1) remove single jumps in the velocity maps by
        averaging over them;
        2) remove Nans from the maps;
        3) average in the horizontal direction;
    Parameters
    ----------
    vel_field : ndarray()
        DESCRIPTION.
    dv : TYPE, optional
        DESCRIPTION. The default is 8.
    dd : TYPE, optional
        DESCRIPTION. The default is 4.
    degree : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    vel_field : TYPE
        DESCRIPTION.

    """
    vel_field = remove_nans(vel_field)

    vel_field = correct_IRIS_noisy_maps(vel_field, dv)

    vel_field = detrend_data(vel_field)
    vel_field = average_velmap_slitwise(vel_field, dd)
    
    return vel_field

def average_every(array, dd=4):
    """
    Rebin down (average) by a factor <dd>
    
    Parameters
    ----------
    array : ndarray [n_steps]
        Input array
    dd : int, optional
        Rebin factor. The default is 4.

    Returns
    -------
    averaged_array : ndarray [n_steps]
        Output array.

    """
    
    bb = [np.median(array[ii*dd:ii*dd+dd]) for ii 
          in range(len(array)//dd-1)]
    averaged_array = np.array(bb)
    
    return averaged_array

def average_velmap_slitwise(vel_map, av_num=4):
    """
    Average a map over the slit dimension. 
    First dimension is slitwise

    Parameters
    ----------
    vel_map : ndarray [n_slit_pixels, n_timesteps]
        Input array

    av_num: int
        Number of pixels the map to be averaged over.

    Returns
    -------
    velmap_average : ndarray [n_slit_pixels/av_num]
        Averaged map over the slit dimension.

    """
    
    vel_map_r = np.swapaxes(vel_map, 0, 1)
    
    vel_map_r_a = np.array([average_every(el, av_num) 
                            for el in vel_map_r])
    
    velmap_average = np.swapaxes(vel_map_r_a, 0, 1)
    return velmap_average

def plot_velocity_map(vel_map, aspect=.2, title="", 
                      d="", vmin=-7.5, vmax=7.5, cadence=17):
    """
    Plot a velocity map.

    Parameters
    ----------
    vel_map : ndarray [n...]
        DESCRIPTION.
    aspect : TYPE, optional
        DESCRIPTION. The default is .2.
    title : TYPE, optional
        DESCRIPTION. The default is "".
    d : TYPE, optional
        DESCRIPTION. The default is "".
    vmin : TYPE, optional
        DESCRIPTION. The default is -7.5.
    vmax : TYPE, optional
        DESCRIPTION. The default is 7.5.
    cadence: float [seconds]
        Cadence of the observations. Scales the y-axis accordingly.

    Returns
    -------
    None.

    """
    
    vel_map = np.swapaxes(vel_map, 0, 1)
    pl.figure(dpi=250)    
    im1 = pl.imshow(vel_map - np.mean(np.ma.masked_invalid(vel_map)), 
                    vmin=vmin, vmax=vmax, cmap="bwr", aspect=aspect, 
                    extent=[0, vel_map.shape[1], 0, vel_map.shape[0]*cadence])
    pl.colorbar(im1, label="Doppler Velocity [km/s]", shrink=1)
    pl.xlabel("Pixels across the slit")
    pl.ylabel("Time [s]")
    pl.title(title)
    
    pl.savefig(d+title+".png", transparent=True)
    pl.show()

def plot_intensity_map(int_map, aspect=.2, vmin=100, vmax=1000, title="", 
                       d="", cadence=17):
    """
    Plot an intensity map out of an array.
    Parameters
    ----------
    int_map :2D ndarray
        2D ndarray with the map.
    aspect : TYPE, optional
        DESCRIPTION. The default is .2.
    vmin : TYPE, optional
        Min intensity. The default is 100.
    vmax : float
        Max Intensity. The default is 1000.
    title : string, optional
        Title for the plot. The default is "".
    d : string
        Directory to save the folder. The default is "".
    cadence : float [seconds]
        Cadence of the observation. Scales the y-axis accordingly.
        The default is 17 s.

    Returns
    -------
    None.

    """
    int_map = np.swapaxes(int_map, 0, 1)
    pl.figure(dpi=250)    
    im1 = pl.imshow(int_map, 
                    vmin=vmin, vmax=vmax, cmap="plasma", aspect=aspect, 
                    extent=[0, int_map.shape[1], 0, int_map.shape[0]*cadence])
    pl.colorbar(im1, label="Intensity [Arbitrary units]", shrink=1)
    pl.xlabel("Pixels across the slit")
    pl.ylabel("Time [s]")
    pl.title(title)
    pl.savefig(d+title+".png", transparent=True)

    pl.show()
    
def plot_Pxx_2D(freq, Pxx, aspect=.2, title="", d="", vmina=-3, vmaxa=.5,
                remove_noise=False, freq_range=[0, 20]):
    """
    Plot the Power spectrum of 2D slice (location along slit vs frequency).

    Parameters
    ----------
    freq : ndarray [Nfreq]
        Frequency
    Pxx : ndarray [Nfreq, nX]
        DESCRIPTION.
    aspect : TYPE, optional
        DESCRIPTION. The default is .2.
    title : str
        Title of the plot. The default is "".
    d : str
        Where the plot to be saved. The default is "".
    vmina : float
        Vmin for the colorscale. The default is -3.
    vmaxa : float
        Vmax for the colorscale. The default is .5.
    remove_noise: bool, optional
        Remove the white noise level based on the last 15 frequency pixels.
        The default is True.
    freq_range: [freq_min, freq_max], optional
        Frequency range to be plotted in mHz. The default is [0, 20] mHz.

    Returns
    -------
    None.

    """
    
    pl.figure(dpi=250)
    Pxx_nx = Pxx.shape[0]
    if remove_noise == True:
        noise_est = np.array([np.mean(el[-5:-1]) for el in Pxx])
        print(noise_est)
        for el in range(Pxx_nx):
            Pxx[el, :] -= noise_est[el] 
    im1 = pl.imshow(np.log10(Pxx).T-3, vmin=vmina, vmax=vmaxa, aspect=aspect,
                    extent=[0, Pxx.shape[0], freq[-1]*1e3, 0])
    pl.xlabel("Pixels along the slit")
    pl.ylabel("Frequency [mHz]")
    
    pl.colorbar(im1, label="Log10[Acoustic power [(km/s)$^2$/mHz]",
                shrink=.8)
    pl.title(title)
    pl.ylim(freq_range[1], freq_range[0])
    pl.savefig(d+title+".png", transparent=True)
    pl.show()

def calc_mask(v_array, dv):
    
    len_x_data = v_array.shape[0]
    
    mask = np.zeros(len_x_data)
    
    for el in range(len_x_data):
        i = 0
        Flag = True
        for i in range(v_array.shape[1]-1):
            if np.abs(v_array[el, i] - v_array[el, i+1]) > dv:
                Flag = False
                break
        mask[el] = Flag
    
    return mask

def calc_Pxx_velmap(vel_map, fsi=1/17):
    """
    Calculate the power spectrum of a map from a velocity time series.

    Parameters
    ----------
    vel_map : 2D ndarray
        DESCRIPTION.
    fsi : float, [1/seconds]
        Sampling frequency. The default is 1/17.

    Returns
    -------
    freq : TYPE
        Frequency grid.
    P_v_container : ndarray
        Power of the velocity signal

    """
    freq_test, Pxx_test = sp.signal.periodogram(vel_map[0, :], fs=fsi)


    P_v_container = np.zeros((vel_map.shape[0], Pxx_test.shape[0])) 

    for el in range(P_v_container.shape[0]):
        freq, P_v_container[el, :] = sp.signal.periodogram(vel_map[el, :], 
                                                           fs=fsi)
       
    P_v_containerm = np.array([np.mean(np.ma.masked_invalid(el).compressed()) 
                            for el in np.swapaxes(P_v_container, 0, 1)])
    
    return freq, P_v_container

def load_data_series(filename, spec_line):
    """
    Load the spectrum for a spec_line from an IRIS lvl 2 data file
    
    Parameters
    ----------
    filename : string
        Filename of the file to be opened
    spec_line : string
        The name of the spec lines to be loaded. Use the following key below:

     TDESC1  = 'C II 1336'          /
     TDESC2  = 'O I 1356'           /
     TDESC3  = 'Si IV 1394'         /
     TDESC4  = 'Si IV 1403'         /
     TDESC5  = '2832    '           /
     TDESC6  = '2814    '           /
     TDESC7  = 'Mg II k 2796'       /          
 
    Returns
    -------
    spectrum : ndarray [ntime, n_dim_slit, n_wave]
        Spectrum of the spectal line specified

    """
    hdu  = fits.open(filename)
    keys = hdu[0].header["TDESC*"]
    
    ii = 1 
    
    while (keys[ii-1] != spec_line):
        ii+=1
    
    spectrum = hdu[ii].data
    header   = hdu[ii].header
    
    return spectrum, header 


def load_velocity_mg2(filename):
    """
    Load and return the velocity and intensity signals from the .sav files 
    produced from the find_mg2_features_lvl2 function:
        
    Parameters
    ----------
    filename : string
        Name of the file

    Returns
    -------
    Legend:
        n_line: == 0, Mg k line
                == 1, Mg h line
    
        n_prop: == 0, Doppler velocity [km/s]
                == 1, Intensity [some units]
            
        n_y: index along the slit
        n_t: time index
        
    bp : ndarray (n_t, n_y, n_prop, n_line)
        Blue wing (2v component) properties 
    rp : ndarray (n_t, n_y, n_prop, n_line)
        Red wing (2r component) properties 
    lc : ndarray (n_t, n_y, n_prop, n_line)
        Line center (3 component) properties .

    """
    
    hdu = readsav(filename)
    rp = hdu["rp"]
    bp = hdu["bp"]
    lc = hdu["lc"]
    
    
    return bp, rp, lc


def correct_IRIS_noisy_series(timeSeries, dd):
    """
    Remove spurious signals in the velocity timeseries

    Parameters
    ----------
    timeSeries : ndarray [n_timesteps]
        DESCRIPTION.
    dd : float
        Threshold over which to interpolate

    Returns
    -------
    timeSeriesCorrected : ndarray [n_timeseries]
        Corrected for the threshold timeseries.

    """
    
    n_timesteps = timeSeries.shape[0]
    n_slitsteps = timeSeries.shape[1]
    
    timeSeriesCorrected = timeSeries * 0.0 + 0.0
    timeSeriesCorrected[0] = timeSeries[0]
    
    
    for el in range(2, n_timesteps-2):
        if (((timeSeries[el+1] - timeSeries[el]) > dd)
            or np.isnan(timeSeries[el+1]) == True):
            timeSeriesCorrected[el+1] = .5*(timeSeries[el] + timeSeries[el+2])
        else:
            timeSeriesCorrected[el+1] = timeSeries[el+1]
    
    return timeSeriesCorrected


def correct_IRIS_noisy_maps(timeSeries, dd):
    """
    Remove spurious signals in the velocity timeseries

    Parameters
    ----------
    timeSeries : ndarray [n_timesteps]
        DESCRIPTION.
    dd : float
        Threshold over which to interpolate

    Returns
    -------
    timeSeriesCorrected : ndarray [n_timeseries]
        Corrected for the threshold difference timeseries.

    """
    
    n_timesteps = timeSeries.shape[0]
    n_slitsteps = timeSeries.shape[1]
    
    timeSeriesCorrected = timeSeries * 0.0 + 0.0
    timeSeriesCorrected[0, :] = timeSeries[0, :]
    
    for xx in range(0, n_slitsteps-1):
        for el in range(0, n_timesteps-2):
            if (((timeSeries[el+1, xx] - timeSeries[el, xx]) > dd)
                or np.isnan(timeSeries[el+1, xx]) == True):
                i = 0
                next_pixel = timeSeries[el+2, xx]
                while np.isnan(next_pixel) == True:
                    i += 1 
                    if (el + 2 + i) >= (n_timesteps-2):
                        next_pixel = np.median(timeSeries[el-10:el-1, xx])
                    else:
                        next_pixel = timeSeries[el+2+i, xx]
                    
                    
                timeSeriesCorrected[el+1, xx] = .5*(timeSeries[el, xx] 
                                                    + next_pixel)
            else:
                timeSeriesCorrected[el+1, xx] = timeSeries[el+1, xx]
    
    for el in range(0, n_timesteps-1):
        for xx in range(0, n_slitsteps-2):
            if (((timeSeries[el, xx+1] - timeSeries[el, xx]) > dd)
                or np.isnan(timeSeries[el, xx+1]) == True):
                i = 0
                next_pixel = timeSeries[el, xx+2]
                while np.isnan(next_pixel) == True:
                    i += 1 
                    if (xx + 2 + i) >= n_slitsteps-2:
                        next_pixel = np.median(timeSeries[el, xx-10:xx-2])
                    else:
                        next_pixel = timeSeries[el, xx+2 + i]
                timeSeriesCorrected[el, xx+1] = .5*(timeSeries[el, xx] 
                                                    + next_pixel)
            else:
                timeSeriesCorrected[el, xx+1] = timeSeries[el, xx+1]
    return timeSeriesCorrected

def detrend_data(timeSeries, dd=2):
    """
    Remove long-term trends from the data by removing a polynomial fit to the 
    data.

    Parameters
    ----------
    timeSeries : 1d ndarray
        Timeseries to be detrended.
    dd : int, optional
        Degree of the polynomial to be removed. The default is 2.

    Returns
    -------
    timeSeries_detrend : 1d ndarray
        Detrended data.

    """
    timeSeries = np.swapaxes(timeSeries, 0, 1)
    dt = 3
    nx = timeSeries.shape[0]
    nt = timeSeries.shape[1]

    
    timeSeries_detrend = np.zeros((nx, nt-dt))
    
    for el in range(nx):
        xx = np.linspace(0, nt-dt-1, num=nt-dt)
        fit_fn = np.poly1d(np.polyfit(xx, timeSeries[el, :-3], dd))
        timeSeries_detrend[el, :] = timeSeries[el, :-dt] - fit_fn(xx)
    
    return timeSeries_detrend
 
