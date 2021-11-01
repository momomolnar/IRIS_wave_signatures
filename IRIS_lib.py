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
from scipy.signal import correlate

import pycwt as wavelet
from pycwt.helpers import find

from RHlib import calc_v_lc

pl.style.use('science')

def calc_median_hist(yedges, n):
    '''
    Calculate the median of a histogram column.

    Parameters
    ----------
    yedges : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    y_edge : TYPE
        DESCRIPTION.

    '''
    y_edge = 0
    n_count = 0
    n_total = np.sum(n)

    while n_count < n_total/2.0:
        n_count = n_count + n[y_edge]
        y_edge = y_edge + 1

    return y_edge

def remove_nans(array, replace="median", print_replacement = False):
    """
    Remove Nans from an array

    Parameters
    ----------
    array : ndarray
        Input array;
    replace: float, Optional
        Replacement for the NaNs; Default is median of the array
    print_replacement: Bool, Optional
        Flag to print the replacement value. Default is False.
    Returns
    -------
    array : ndarray
        Filtered array;

    """
    array1 = array.copy()
    if replace=="median":
        replace = np.mean(np.ma.masked_invalid(array))
    if print_replacement == True:
        print(f"Replacement for the array is: {replace}")
    for ii in range(array.shape[0]):
        for jj in range(array.shape[1]):
            if np.isnan(array[ii, jj]) == True:
                array1[ii, jj] = replace
    
    return array1

def filter_velocity_field(vel_field, dt=8, dd=4, dx=10, degree=2):
    """
    Filter raw velocity map from IDL input
    by doing the following three steps:
        0) Replace NaNs with something very small and then remove those 
        really small numbers with the median of 3x3 kernel
        1) remove single jumps in the velocity maps by
        averaging over them;
        2) Remove longterm trends along the temporal direction without
        by removing a polynomial of degree <degree>.
        3) average in the horizontal (slit) direction;
    
    Parameters
    ----------
    vel_field : ndarray()
        Input velocity field to be filtered.
    dt : int, optional, The default is 8.
        Velocity jump over which to normalize in the temporal direction.
    dx : int, optional,  The default is 10
        Velocity jump over which to normaliza in the slit direction.
    
    dd : int, optional
        Averaging over the slit dimension. The default is 4.
    degree : int, optional
        Degree of the polynomial to be removed from the data. The default is 2.

    Returns
    -------
    vel_field : TYPE
        DESCRIPTION.

    """
    
    filter_value = -30
    vel_field = np.array(vel_field)
    
    vel_field[np.isnan(vel_field)] = filter_value
    vel_field = remove_nanvalues(vel_field, filter_value=filter_value)

    vel_field = correct_IRIS_noisy_maps(vel_field, dx=dx, dt=dt)


    vel_field = average_velmap_slitwise(vel_field, dd)
    vel_field = detrend_data(vel_field).T

    return vel_field.T

def remove_nanvalues(vel_field, filter_value=-30):
    """
    Remove the very negative numbers from our array that were NaNs 

    Parameters
    ----------
    vel_field : ndarray [nX, nY]
        Velocity field to be filtered. 
    filter_value: int
        Negative value to be removed from the array

    Returns
    -------
    vel_field : ndarray [nX, nY]
        Filtered velocity field

    """
    
    for ii in range(4, vel_field.shape[0]-4):
        for jj in range(4, vel_field.shape[1]-4):
            if vel_field[ii, jj] <= filter_value:
                vel_field[ii,jj] = 0
                X = np.ma.masked_equal(vel_field[ii-4:ii+4, jj-4:jj+4], filter_value)
                vel_field[ii, jj] = np.median(X.compressed())
    
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
    
    bb = [np.median(array[ii:(ii+dd)]) for ii 
          in range(len(array)-1)]
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
                      d="", vmin=-7.5, vmax=7.5, cadence=17, cmap="bwr"):
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
    cmap: string, optional
        Colormap to be used. Default is "bwr".
    Returns
    -------
    None.

    """
    
    vel_map = np.swapaxes(vel_map, 0, 1)
    pl.figure(dpi=250)    
    im1 = pl.imshow(vel_map - np.mean(np.ma.masked_invalid(vel_map)), 
                    vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect, 
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
                remove_noise=False, freq_range=[]):
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
        Frequency range to be plotted in mHz. The default is no frequency range.

    Returns
    -------
    None.

    """
    
    pl.figure(dpi=250)
    Pxx_nx = Pxx.shape[0]
    if remove_noise == True:
        noise_est = np.array([np.amin(el[-5:-1]) for el in Pxx])
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
    # pl.ylim(freq_range[1], freq_range[0])
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
        Input data.
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
        if ((np.abs(timeSeries[el+1] - timeSeries[el]) > dd)
            or np.isnan(timeSeries[el+1]) == True):
            timeSeriesCorrected[el+1] = timeSeries[el]
                                        #2*timeSeries[el] - timeSeries[el-1]
                                        #.5*(timeSeries[el] + timeSeries[el+2])
        else:
            timeSeriesCorrected[el+1] = timeSeries[el+1]
    
    return timeSeriesCorrected


def correct_IRIS_noisy_maps(timeSeries, dt=15, dx=15):
    """
    Remove spurious signals in the velocity timeseries

    Parameters
    ----------
    timeSeries : ndarray [n_timesteps]
        DESCRIPTION.
    dt : float
        Threshold over which to interpolate over in the temporal dimension.
    dx : float     
        Threshold over which to interpolate over in the slit dimension.
    
     

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
            if (np.abs(timeSeries[el+1, xx] - timeSeriesCorrected[el, xx]) > dt):                    
                timeSeriesCorrected[el+1, xx] = np.median(timeSeries[el-3:el+3, xx-3:xx+3])
            else:
                timeSeriesCorrected[el+1, xx] = timeSeries[el+1, xx]
    
    for el in range(0, n_timesteps-1):
        for xx in range(0, n_slitsteps-2):
            if (np.abs(timeSeries[el, xx+1] - timeSeriesCorrected[el, xx]) > dx):
                timeSeriesCorrected[el, xx+1] = np.median(timeSeries[el-3:el+3, xx-3:xx+3])
                                                #- timeSeriesCorrected[el-1, xx])
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
        try:
            fit_fn = np.poly1d(np.polyfit(xx, timeSeries[el, :-3], dd))
            timeSeries_detrend[el, :] = timeSeries[el, :-dt] - fit_fn(xx)
        except:
            mean_timeSeries = np.nanquantile(timeSeries[el, :], 0.5)
            timeSeries_detrend[el, :] -= mean_timeSeries
    return timeSeries_detrend
 
def find_lag(timeSeries1, timeSeries2):
    """
    Calculate what is the lag between two time series based on their xcorrela-
    tion (computed with Scipy)

    Parameters
    ----------
    timeSeries1 : ndarray [Nt] 
        Timeseries 1
    timeSeries2 : ndarray [Nt]
        Timeseries 2

    Returns
    -------
    time_lag : float
        Lag between the two timeseries [measured in indices]

    """
    
    cor = correlate(timeSeries1, timeSeries2, mode="same")
    lag = np.argmax(cor)
    time_lag = lag - len(timeSeries1)/2
    
    return time_lag

def convert_cart_to_polar(x, y):
    """
    Convert Cartesian [x,y] to polar coordinates [r, theta]

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.

    """
    
    r     = np.sqrt(x*x + y*y)
    if y > 0:
        theta = np.arccos(x/r)
    else:
        theta = -1 * np.arccos(x/r)
  
    return r, theta

def convert_polar_to_cart(coords):
    x = coords[0] * np.cos(coords[1])
    y = coords[0] * np.sin(coords[1])

    return [x, y]


def angle_mean(angles):
    """
    Calculate the mean of a distribution of angles
    using the unit circle averaging.
    
    Parameters
    ----------
    angles : ndarray [num_angles]
        Input Angles array.

    Returns
    -------
    mean_angle : float
        The mean angle.

    """
    
    num_angles = len(angles)
    r = np.ones(num_angles)
    
    xy_array = [convert_polar_to_cart(el) for el in 
                zip(r, angles)]
    
    xy_array = np.array(xy_array)
    try:
        x_mean = np.mean(xy_array[:, 0])
        y_mean = np.mean(xy_array[:, 1])
    
        r, mean_angle = convert_cart_to_polar(x_mean, y_mean)
    except: 
        mean_angle = 0
    return mean_angle


def calc_phase_diff(timeSeries1, timeSeries2, dt=16.7, t0=0, 
                    use_cache=False):
    """
    Compute the phase between two signals (timeSeries1 and timeSeries2) based 
    on their coherence on Morlet wavelet analysis. Tailored for IRIS level 2 
    data.

    Parameters
    ----------
    timeSeries1 : ndarray [num_tSteps]
        Timeseries 1
    timeSeries2 : ndarray [num_tSteps]
        Timeseries 2
    dt : float, optional
        Cadence of the data. The default is 16.7 [s].
    t0 : float, optional
        Starting time of the data. The default is 0.
    use_cache: bool, optional
        Use cached estimate for the red noise level on the first iteration. 
        Default set to False, as for new data series you have to model the 
        red noise level separately.

    Returns
    -------
    angles : TYPE
        DESCRIPTION.
    cor_sig_all : TYPE
        DESCRIPTION.
    freq : TYPE
        DESCRIPTION.

    """

    angles = np.zeros((72, 123, timeSeries1.shape[0]))
    cor_sig_all = np.zeros((72, 123, timeSeries1.shape[0]))

    for el in range(0, timeSeries1.shape[0]-2):
        print(el)
        N = timeSeries1.shape[1]
        t = np.arange(0, N) * dt + t0

        timeSeries1_ex = timeSeries1[el, :]
        std_timeSeries1 = timeSeries1_ex.std()  # Standard deviation
        var_timeSeries1 = std_timeSeries1 ** 2  # Variance
        timeSeries1_norm = timeSeries1_ex / std_timeSeries1  # Normalized dataset

        timeSeries2_ex = timeSeries2[el, :]
        std_timeSeries2 = timeSeries2_ex.std()  # Standard deviation
        var_timeSeries2 = std_timeSeries2 ** 2  # Variance
        timeSeries2_norm =timeSeries2_ex / std_timeSeries2  # Normalized dataset

        t1 = t2 = np.linspace(0, timeSeries1_ex.size - 1, 
                              num=timeSeries1_ex.size)*dt

        mother = wavelet.Morlet(6)
        s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 secondss = 6 months
        dj = 1 / 12  # Twelve sub-octaves per octaves
        J = 7 / dj  # Seven powers of two with dj sub-octaves
        print(timeSeries1_ex.shape)
        try:
            alpha, _, _ = wavelet.ar1(timeSeries1_norm)  # Lag-1 autocorrelation for red noise


            WCT, aWCT, cor_coi, freq, sig = wavelet.wct(timeSeries1_norm, 
                                                         timeSeries2_norm, 
                                                         dt, dj=1/12, 
                                                         s0=s0, J=-1,
                                                         significance_level=0.8646,
                                                         wavelet='morlet', 
                                                         normalize=True,
                                                         cache=use_cache)

            cor_sig = np.ones([1, timeSeries1_ex.size]) * sig[:, None]
            cor_sig = np.abs(WCT) / cor_sig  # Power is significant where ratio > 1
            cor_sig_all[:, :, el] = cor_sig
            use_cache=True
            angles[:, :, el] = aWCT
            
            coi_mask = make_coi_mask(freq, cor_coi)
        except:
             print(f"Iteration {el} didn't find an AR model to fit the data")
    # Calculates the phase between both time series. The phase arrows in the
    # cross wavelet power spectrum rotate clockwise with 'north' origin.
    # The relative phase relationship convention is the same as adopted
    # by Torrence and Webster (1999), where in phase signals point
    # upwards (N), anti-phase signals point downwards (S). If X leads Y,
    # arrows point to the right (E) and if X lags Y, arrow points to the
    # left (W).
            

    return angles, cor_sig_all, freq, coi_mask


def make_coi_mask(freq, cor_coi):
    """
    Make a mask for the cone of influence that can be applied to the measured 
    quantities from the 

    Parameters
    ----------
    freq : ndarray [num_freq]
        Frequency array
    cor_coi : ndarray [num_time]
        Cone of influence 

    Returns
    -------
    coi_mask : ndarray [num_freq, num_time]
        Cone of influence mask to be applied.

    """
    num_freq = freq.size
    num_time = cor_coi.size 
    
    coi_mask = np.zeros((num_freq, num_time))
    
    for ii in range(num_time):
        min_freq = 1/cor_coi[ii]
        jj = 0
        while freq[jj] > min_freq:
            coi_mask[jj, ii] = 1
            jj += 1
    
    return coi_mask

def filter_CII_data(spectral_map, kernel_size=3):
    '''
    Filter out a dopplergram following the Rathore and Carlsson, 2015, ApJ, 
    method. Intended to be used for filtering out the C II IRIS rasters. 
    
    s_filt = 1) if sigma_s > sigma -> (sigma/sigma_s)^2 * m_s 
                                      + (1 - (sigma/sigma_s)^2) * m_s
             2) if sigma_s < sigma -> m_s
    

    Parameters
    ----------
    spectral_map : ndarray [numSlitLoc, numWaves]
        Raw spectral map input.
    kernel_size: int 
        Kernel size for the averaging window. 
    Returns
    -------
    spectral_map_filted : ndarray [numSlitLoc, numWaves]
        Filtered spectral map output.
    '''
    
    d_x = (kernel_size - 1) // 2                        #  kernel halfwidth 
    size_X = spectral_map.shape[0]
    size_Y = spectral_map.shape[1]
    
    mean_spectrum = np.array([[np.mean(spectral_map[i-d_x:i+d_x, j-d_x:j+d_x]) 
                               for i in range(1, size_X-1)] 
                               for j in range(1, size_Y-1)]).T
    
    var_spectrum = np.array([[np.var(spectral_map[i-d_x:i+d_x, j-d_x:j+d_x]) 
                              for i in range(1, size_X-1)] 
                              for j in range(1, size_Y-1)]).T
    var_ave = np.mean(var_spectrum)
    
    velocity_map_filter = np.zeros((size_X, size_Y))
    
    var_norm = var_ave / var_spectrum
    
    velocity_map_filter = (var_norm * mean_spectrum 
                           + (1 - var_norm) * spectral_map[1:-1, 1:-1])
    low_SNR_region = np.where(var_spectrum < var_ave)
    velocity_map_filter[low_SNR_region] = mean_spectrum[low_SNR_region]
    
    return velocity_map_filter