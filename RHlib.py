#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:47:55 2019

@author: momo
"""

import numpy as np
import scipy as sp
import radynpy as rp
import matplotlib.pyplot as plt
import astropy.io.fits as fits


from scipy import interpolate as interpolate
from scipy import optimize as optimize
from scipy import integrate as integrate
from scipy.io import readsav
from scipy import signal

# Some global (physical) constansts
c = 3e5  # km/s
ca_coeff = 1233.45
ha_coeff = 2089.
h_mass   = 1.6735e-27  # kg


def line_intensity_with_cont(rad, kr, muIdx):
    """
    Return the line profile in physical units (cgs) and Å
    """

    if not not rad.cont[kr]:
        print('line_intensity cannot compute bf intensity')
        return 1

    wl = rad.alamb[kr] / (rad.q[0:rad.nq[kr], kr] *  rad.qnorm * 1e5 / rad.cc + 1)
    intens = (rad.outint[:, 1:rad.nq[kr]+1, muIdx, kr]
        + rad.outint[:, 0, muIdx, kr][:, np.newaxis]) * rad.cc * 1e8 / (wl**2)[np.newaxis, :]
    return wl[::-1], intens[:,::-1]


def parabola(x, coeffs):
    return [coeffs[0] * el**2 + coeffs[1] * el + coeffs[2] for el in x]


def calc_power_coeff(freqs, wave):
    """
    Calculate the power conversion coefficient from IBIS velocity files
    to km^2/s^2 / Hz.
    """


def calc_v_cog(wave, I):
    '''
    Copy of the IDL version for calculating the COG velocity
    '''
    I = I[0] - I
    dwave = wave[1] - wave[0]
    numer = np.sum(wave*I)
    denom = np.sum(I)
    return numer/denom

def calc_v_lc(wave,I,ns,nf,num):
    '''
    Calculate the velocity from the line center as in the vel_lc IDL
    routine found in SolarSoft by fitting a parabola to the points around
    the line core

    Parameters
    ----------
    wave : ndarray (n_wave)
        Wavelength grid
    I : ndarray (n_waves)
        Intensity grid
    ns : int
        point to start looking for a center
    nf : int (negative )
         point to end looking for a center
    num : int
        number of points to be fit to the parabola

    Returns
    -------
    cent : center of the line from intensity
    '''
    numh = int((num-1)/2)
    cent = np.where(I == I[ns:nf].min())
    cent = cent[0][0] + ns
    del1 = I[(cent-numh):(cent+numh)]
    coeff = np.polyfit(wave[cent-numh:cent+numh], del1, deg=2)
    cent = -1 * coeff[1] / (2.*coeff[0])

    return cent, coeff


def calc_A_conserve_F(Amplitude, Atmos, x, rho):
    '''
    Calculate the amplitude of a wave propagating under conservation of
    energy flux assuming iso-sound of speed-like atmosphere.
    '''
    rho_0 = rho
    rho_c = np.sum(Atmos.nH[:, x])
    return Amplitude * np.sqrt(rho_0 / rho_c)


def write_fits(data, hdr_words, write_path):
    '''

    Parameters
    ----------
    data : n-dimensional array
        Data to be written in the FITS file
    hdr_words : dictionary
        Dictionary with the header keywords

    Returns
    -------
    str
        Succesfully written fits file

    '''

    hdr = fits.Header()

    for keys in hdr_words.keys():
        hdr[keys] = hdr_words[keys]
    fits_file = fits.PrimaryHDU(data = data,header = hdr)
    fits_file.writeto(write_path)

    return 'Successfully written FITS file'

class Atmos_height():
    ''' Class to hold all the atmospheric info
    numpy.interp(x, xp, fp,...)'''

    def __init__(self, dataDir, fileName, Nx_num_new):

        self.model_name = fileName
        fileName1 = dataDir + fileName
        atmos_temp = (np.loadtxt(fileName1)).transpose()
        nH_temp    = (np.loadtxt(fileName1+'_pops')).transpose()
        self.cs    = 7 #speed of sound in the atmosphere in km/s
        self.Nx_num_original = (atmos_temp.shape)[1]

        if Nx_num_new == 0:
            self.Nx_num_new = self.Nx_num_original
        else:
            self.Nx_num_new = Nx_num_new

        self.Nx      = list(range(self.Nx_num_original))
        self.Nx_new  = list(range(self.Nx_num_new))

        height_new   = np.linspace(atmos_temp[0,-1],
                                   atmos_temp[0,0], num=self.Nx_num_new)
        self.height  = height_new
        h            = np.flip(atmos_temp[0, :])

        self.H_levels   = 6
        self.model_name = fileName
        self.T          = np.interp(height_new, h,
                                     np.flip(atmos_temp[1,:]))
        #print(atmos_temp[0,:])
        #print(atmos_temp[1,:])
        #print(height_new)
        #
        #plt.figure(figsize=(10,8))
        #plt.plot(self.height,self.T,'y.',label='Interpolated Temperature')
        #plt.plot(h,atmos_temp[1,:],'b--',label='initial Temperature')
        #plt.legend()
        #plt.show()

        self.Ne          = np.interp(height_new, h,
                                     np.flip(atmos_temp[2, :]))
        self.V           = np.interp(height_new, h,
                                     np.flip(atmos_temp[3, :]))
        self.Vturb       = np.interp(height_new, h,
                                     np.flip(atmos_temp[4, :]))
        self.B           = np.zeros(self.Nx_num_new)
        self.phi         = np.zeros(self.Nx_num_new)
        self.gamma       = np.zeros(self.Nx_num_new)
        self.nH          = np.zeros((self.H_levels, self.Nx_num_new))

        for i in range(self.H_levels):
            self.nH[i, :] = np.interp(height_new, h, np.flip(nH_temp[i, :]))
        print('Atmosphere loaded')

    def plot_v_field(self):
        plt.plot(self.height, self.V, 'r.--')
        plt.ylabel('Vertical Velocity, [km/s]')
        plt.xlabel('Height, [km]')
        plt.grid(alpha=0.5)
        plt.show()

    def plot_T(self):
        plt.plot(self.height, self.T, 'r.--')
        plt.ylabel('Temperature, [K]')
        plt.xlabel('Height, [km]')
        plt.grid(alpha=0.5)
        plt.yscale('log')
        plt.show()

    def plot_Ne(self):
        plt.plot(self.height, self.Ne, 'r.--')
        plt.ylabel('Ne, cm-3')
        plt.xlabel('Height, [km]')
        plt.grid(alpha=0.5)
        plt.yscale('log')
        plt.show()

    def include_V_wave(self, nu, phi, Amplitude):
        '''
        For now with constant amplitude; to implement changing
        amplitude to conserve flux
        '''

        self.V = Amplitude * np.sin(2 * np.pi * nu * self.height / self.cs
                                    + phi)

    def include_V_shock(self, nu, phi, Amplitude, rho):
        '''
        Propagate the shock wave as if the flux is conserved
        rho(x) * v_c * v_perp ** 2  = const
        '''
        Amplitude_shock = [ calc_A_conserve_F(Amplitude, self, i, rho)
                           for i in self.Nx_new]
        self.V = Amplitude_shock * np.sin(2 * np.pi * nu
                                          * self.height / self.cs + phi)


    def include_V_pulse(self, index, Amplitude):
        self.V = np.zeros(self.Nx_num_new)
        self.V[index] = Amplitude


    def write_TF_test(self, nu_min, nu_max, nu_num, phi_num, Amplitude, wave,
                       write_Ne):
        ''' Write an RH output atmosphere for Ricardos's version of RH
        Format has to be the following:
        Height,Temperature,Ne,V,Vturb,field,gamma,phi,nh(1),nh(2),nh(3),nh(4),
        nh(5),np
        '''

        hdr_dic = {}
        hdr_dic['Nu_num'] = nu_num
        hdr_dic['Nu_min'] = nu_min
        hdr_dic['Nu_max'] = nu_max
        hdr_dic['Phi_num'] = phi_num
        hdr_dic['Amp'] = Amplitude * 1e3
        hdr_dic['Write_Ne'] = write_Ne
        hdr_dic['Wave'] = wave

        rho = 1e12
        hdr_dic['Rho_0'] = rho * h_mass * 1e6

        if write_Ne is True:
            atmos = np.zeros((phi_num, nu_num, 14, self.Nx_num_new))
        else:
            atmos = np.zeros((phi_num, nu_num, 8, self.Nx_num_new))

        nu    = np.logspace(np.log10(nu_min), np.log10(nu_max), num=nu_num)

        for jj in range(nu_num):
            for ii in range(phi_num):
                atmos[ii, jj, 0, :] = np.flip(self.height)
                atmos[ii, jj, 1, :] = np.flip(self.T)
                atmos[ii, jj, 2, :] = np.flip(self.Ne)
                atmos[ii, jj, 4, :] = np.flip(self.Vturb)
                atmos[ii, jj, 5, :] = np.flip(self.B)
                atmos[ii, jj, 6, :] = np.flip(self.phi)
                atmos[ii, jj, 7, :] = np.flip(self.gamma)

                if write_Ne is True:
                    for kk in range(int(self.H_levels)):
                        atmos[ii, jj, (8+kk), :] = np.flip(self.nH[kk, :])

                self.include_V_shock(nu[jj], ii*2*np.pi/phi_num, Amplitude, rho)
                atmos[ii, jj, 3, :] = self.V

        atmos = np.swapaxes(atmos, 1, 2)
        atmos = np.swapaxes(atmos, 0, 3)

        write_path = ('/Users/molnarad/Desktop/rh/' + self.model_name
                      + '_nu_' + str(nu_num) + '_phi_num_' + str(phi_num)
                      + '_A_' + str(Amplitude) + '.fits')
        write_fits(atmos, hdr_dic, write_path)


        print('Successfully written out atmosphere')

    def write_response_test(self, Amplitude, write_Ne):
        ''' Write an RH output atmosphere for Ricardos's version
        of RH

        Format has to be the following:
        Height,Temperature,Ne,V,Vturb,field,gamma,phi,nh(1),nh(2),nh(3),nh(4),nh(5),np
        '''


        if write_Ne is True:
            atmos = np.zeros((1, self.Nx_num_new, 14, self.Nx_num_new))
        else:
            atmos = np.zeros((1, self.Nx_num_new, 8, self.Nx_num_new))

        for jj in range(self.Nx_num_new):
            atmos[0, jj, 0, :] = np.flip(self.height)
            atmos[0, jj, 1, :] = np.flip(self.T)
            atmos[0, jj, 2, :] = np.flip(self.Ne)
            atmos[0, jj, 4, :] = np.flip(self.Vturb)
            atmos[0, jj, 5, :] = np.flip(self.B)
            atmos[0, jj, 6, :] = np.flip(self.phi)
            atmos[0, jj, 7, :] = np.flip(self.gamma)
            if write_Ne is True:
                atmos[0, jj, 8:14, :] = np.flip(self.nH)
            self.include_V_pulse(jj, Amplitude)
            atmos[0, jj, 3, :] = np.flip(self.V)

        atmos = np.swapaxes(atmos, 1, 2)
        atmos = np.swapaxes(atmos, 0, 3)
        hdu = fits.PrimaryHDU(atmos)
        hdu.writeto('/Users/molnarad/Desktop/rh/' + self.model_name
                    + '_RF_vel_'
                    + str(Amplitude) + '_kms.fits', overwrite=True)

        print('Successfully written out atmosphere')

class Atmos_cmass():
    ''' Class to hold all the atmospheric info
    numpy.interp(x, xp, fp,...)'''

    def __init__(self, fileName, Nx_num_new):

        atmos_temp = (np.loadtxt(fileName)).transpose()
        nH_temp    = (np.loadtxt(fileName+'_pops')).transpose()
        self.cs    = 7 #speed of sound in the atmosphere in km/s
        self.Nx_num_original = (atmos_temp.shape)[1]

        if Nx_num_new == 0:
            self.Nx_num_new = self.Nx_num_original
        else:
            self.Nx_num_new = Nx_num_new

        self.Nx      = list(range(self.Nx_num_original))
        self.Nx_new  = list(range(self.Nx_num_new))

        height_new   = np.linspace(atmos_temp[0,0],
                                   atmos_temp[0,-1], num=self.Nx_num_new)

        self.height  = height_new
        h            = atmos_temp[0, :]

        self.H_levels   = 6
        self.model_name = fileName
        self.T          = np.interp(height_new, h,
                                     atmos_temp[1,:])
        #print(atmos_temp[0,:])
        #print(atmos_temp[1,:])
        #print(height_new)
        #
        #plt.figure(figsize=(10,8))
        #plt.plot(self.height,self.T,'y.',label='Interpolated Temperature')
        #plt.plot(h,atmos_temp[1,:],'b--',label='initial Temperature')
        #plt.legend()
        #plt.show()

        self.Ne          = np.interp(height_new, h,
                                     (atmos_temp[2, :]))
        self.V           = np.interp(height_new, h,
                                     (atmos_temp[3, :]))
        self.Vturb       = np.interp(height_new, h,
                                     (atmos_temp[4, :]))
        self.B           = np.zeros(self.Nx_num_new)
        self.phi         = np.zeros(self.Nx_num_new)
        self.gamma       = np.zeros(self.Nx_num_new)
        self.nH          = np.zeros((self.H_levels, self.Nx_num_new))

        for i in range(self.H_levels):
            self.nH[i, :] = np.interp(height_new, h, (nH_temp[i, :]))
        print('Atmosphere loaded')

    def plot_v_field(self):

        plt.plot(self.height, self.V, 'r.--')
        plt.ylabel('Vertical Velocity, [km/s]')
        plt.xlabel('Height, [km]')
        plt.grid(alpha=0.5)
        plt.show()

    def plot_T(self):
        plt.plot(self.height, self.T, 'r.--')
        plt.ylabel('Temperature, [K]')
        plt.xlabel('Height, [km]')
        plt.grid(alpha=0.5)
        plt.yscale('log')
        plt.show()

    def plot_Ne(self):
        plt.plot(self.height, self.Ne, 'r.--')
        plt.ylabel('Ne, cm-3')
        plt.xlabel('Height, [km]')
        plt.grid(alpha=0.5)
        plt.yscale('log')
        plt.show()

    def include_V_wave(self, nu, phi, Amplitude):
        '''
        For now with constant amplitude; to implement changing
        amplitude to conserve flux
        '''
        self.V = Amplitude * np.sin(2 * np.pi * nu * self.height / self.cs
                                    + phi)

    def include_V_pulse(self, index, Amplitude):
        self.V = np.zeros(self.Nx_num_new)
        self.V[index] = Amplitude

    def write_TF_test(self, nu_min, nu_max, nu_num, phi_num, Amplitude,
                       write_Ne):
        ''' Write an RH output atmosphere for Ricardos's version
        of RH
        Format has to be the following:
        Height,Temperature,Ne,V,Vturb,field,gamma,phi,nh(1),nh(2),nh(3),nh(4),nh(5),np
        '''
        if write_Ne is True:
            atmos = np.zeros((phi_num, nu_num, 14, self.Nx_num_new))
        else:
            atmos = np.zeros((phi_num, nu_num, 14, self.Nx_num_new))

        nu    = np.logspace(np.log10(nu_min), np.log10(nu_max), num=nu_num)

        for jj in range(nu_num):
            for ii in range(phi_num):
                atmos[ii, jj, 0, :] = (self.height)
                atmos[ii, jj, 1, :] = (self.T)
                atmos[ii, jj, 2, :] = (self.Ne)
                atmos[ii, jj, 4, :] = (self.Vturb)
                atmos[ii, jj, 5, :] = (self.B)
                atmos[ii, jj, 6, :] = (self.phi)
                atmos[ii, jj, 7, :] = (self.gamma)

                if write_Ne is True:
                    for kk in range(int(self.H_levels)):
                        atmos[ii, jj, (8+kk), :] = (self.nH[kk, :])

                if write_Ne is False:
                    for kk in range(int(self.H_levels)):
                        atmos[ii, jj, (8+kk), :] = 0

                self.include_V_wave(nu[jj], ii*2*np.pi/phi_num, Amplitude)
                atmos[ii, jj, 3, :] = self.V

        atmos = np.swapaxes(atmos, 1, 2)
        atmos = np.swapaxes(atmos, 0, 3)

        hdu = fits.PrimaryHDU(atmos)
        self.model_name = 'FALC_93'
        hdu.writeto('/Users/molnarad/Desktop/rh/'
                    + self.model_name + '_nu_' + str(nu_num) +
                    '_phi_num_' + str(phi_num) + '_A_' + str(Amplitude)
                    + '.fits', overwrite=True)

        print('Successfully written out atmosphere')

    def write_response_test(self, Amplitude, write_Ne):
        ''' Write an RH output atmosphere for Ricardos's version
        of RH

        Format has to be the following:
        Height,Temperature,Ne,V,Vturb,field,gamma,phi,nh(1),nh(2),nh(3),nh(4),nh(5),np
        '''
        if write_Ne is True:
            atmos = np.zeros((1, self.Nx_num_new, 14, self.Nx_num_new))
        else:
            atmos = np.zeros((1, self.Nx_num_new, 14, self.Nx_num_new))

        for jj in range(self.Nx_num_new):
            atmos[0, jj, 0, :] = np.flip(self.height)
            atmos[0, jj, 1, :] = np.flip(self.T)
            atmos[0, jj, 2, :] = np.flip(self.Ne)
            atmos[0, jj, 4, :] = np.flip(self.Vturb)
            atmos[0, jj, 5, :] = np.flip(self.B)
            atmos[0, jj, 6, :] = np.flip(self.phi)
            atmos[0, jj, 7, :] = np.flip(self.gamma)
            if write_Ne is True:
                atmos[0, jj, 8:13, :] = np.flip(self.nH[0:5, :])

            self.include_V_pulse(jj, Amplitude)
            atmos[0, jj, 3, :] = np.flip(self.V)

        atmos = np.swapaxes(atmos, 1, 2)
        atmos = np.swapaxes(atmos, 0, 3)
        hdu = fits.PrimaryHDU(atmos)
        self.model_name = 'FALC_93'
        hdu.writeto('/Users/molnarad/Desktop/rh/' + self.model_name
                    + '_RF_vel_'
                    + str(Amplitude) + '_kms.fits', overwrite=True)

        print('Successfully written out atmosphere')


class RF_vel_calibration():

    def __init__(self, Filename, Amplitude, wave):
        data            = fits.open(Filename)
        data            = data[0].data
        self.data       = data
        self.wavelength = data[0, :, 0, 0]
        self.Amplitude  = Amplitude
        self.Nx         = data.shape[2]
        self.spectra    = data[1, :, :, :]
        self.vel_lc     = data[1, 0, :, :]
        self.wave       = wave

        if wave == 656.3:
            self.Ha_waves = [225, -40]
        if wave == 854.2:
            self.Ha_waves = [340, -55]

        self.Ha_waves_u, self.unique_i = np.unique(
                self.wavelength[self.Ha_waves[0]:(self.Ha_waves[1])],
                return_index=True)
        self.Ha_I = (self.spectra[self.Ha_waves[0]:(self.Ha_waves[1]),
                                  :, :])[self.unique_i, :, :]
        self.N_waves_Ha = len(self.unique_i)

    def Instrument_degrade(self, Instrument):
        # First Convolve spectrum with SPSF
        # Second Average the profile in time

        exp_time = Instrument.sampling_rate
        self.Ha_degraded = np.zeros((self.N_waves_Ha, self.N_nu, self.N_phi))

        for ii in range(self.N_nu):
            nu1       = self.nu[ii]
            N_average = int(self.N_phi * nu1 * exp_time)

            if N_average == 0:
                N_average = 1

            weights = np.ones(N_average) / N_average
            # print('For Frequency %f the degradation is over %d frames'
            #       % (nu1, N_average))
            for ll in range(self.N_waves_Ha):
                self.Ha_degraded[ll, ii, :] = np.convolve(self.Ha_I[ll, ii, :],
                                weights, mode='same')

    def find_vel_RF(self, Plot_B):
        ''' Calculate the line center with non-degraded
        spectral profiles
        '''
        yy = 0
        for xx in range(self.Nx):
            I_rf = interpolate.CubicSpline(self.Ha_waves_u,
                                           self.Ha_I[:, xx, yy])
            self.vel_lc[xx, yy] = (optimize.fmin(I_rf, self.wave, disp=False) -
                       self.wave) / self.wave * c

            if (Plot_B is True):
                plt.plot(self.Ha_waves_u, self.Ha_I[:, xx, yy], 'b.')
                plt.plot(np.tile(self.vel_lc[xx, yy],
                                 len(self.Ha_waves_u)), self.Ha_I[:, xx, yy])
                plt.xlim(self.wave-.1, self.wave + .1)
                plt.xlabel('Wavelength, [nm]')
                plt.show()

            #self.vel_lc[xx, :] = self.vel_lc[xx, :] - np.mean(self.vel_lc[xx, :])
        return 0

    def calculate_RF(self, Plot):
        self.find_vel_RF(Plot)
        self.RF = np.zeros(self.Nx)
        for ii in range(self.Nx):
            self.RF[ii] = self.vel_lc[ii]/self.Amplitude
        return 0


class Spectral_Line():


    def __init__(self, dataDir, modelFilename, specFilename, wave):

        '''


        Parameters
        ----------
        dataDir : string
            Where the Ricardo's RH is installed
        modelFilename : string
            Model Atmosphere Filename
        specFilename : string
            Synthesized spectrum filename

        Returns
        -------
        None.

        '''

        print(dataDir + 'results/' + specFilename)
        hdu            = fits.open(dataDir + 'results/' + specFilename)
        modelFile      = fits.open(dataDir + modelFilename)

        hdr_dic = modelFile[0].header
        #print(hdr_dic)
        nu_min = hdr_dic['Nu_min']
        nu_max = hdr_dic['Nu_max']
        nu_num = hdr_dic['Nu_num']
        phi_num = hdr_dic['Phi_num']
        Amplitude = hdr_dic['Amp']

        rho_0 = hdr_dic['Rho_0']

        data            = hdu[0].data
        self.wavelength = data[0, :, 0, 0]
        self.nu         = np.logspace(np.log10(nu_min),
                                      np.log10(nu_max), num=nu_num)
        self.phi        = np.linspace(0, 2*np.pi, num=phi_num)
        self.Amplitude  = Amplitude
        self.spectra    = data[1, :, :, :]
        self.vel_lc     = data[1, 0, :, :]
        self.width      = data[1, 0, :, :]
        self.vel_gr     = data[1, 0, :, :]
        self.N_nu       = (data.shape)[2]
        self.N_phi      = (data.shape)[3]
        if wave == 656.3:
            self.Ha_waves = [193, -5]
        if wave == 854.2:
            self.Ha_waves = [240, -55]
        self.wave       = wave
        self.nu_ac      = 5e-3
        self.cs         = 7e3 # m/s
        self.rho_0      = rho_0

        self.Ha_waves_u, self.unique_i = np.unique(
                self.wavelength[self.Ha_waves[0]:self.Ha_waves[1]],
                return_index=True)
        self.Ha_I       = (self.spectra[self.Ha_waves[0]:(self.Ha_waves[1]),:,:])[self.unique_i,:,:]
        self.N_waves_Ha = len(self.unique_i)

    def Instrument_degrade(self, Instrument, verbose=False):
        # First sample the line on the wavelength grid of the instrument
        # which for IBIS usually is 0.05 Å.
        # Second Average the profile in time

        exp_time = Instrument.sampling_rate



        self.Ha_degraded = np.zeros((self.N_waves_Ha,
                                     self.N_nu, self.N_phi))


        for ii in range(self.N_nu):
            nu1               = self.nu[ii]
            N_average         = int(self.N_phi * nu1 * exp_time)

            if N_average == 0:
                N_average = 1

            weights = np.ones(N_average) / N_average
            if verbose==True:
                print('For Frequency %f the degradation is over %d frames'
                      % (nu1, N_average))
            for ll in range(self.N_waves_Ha):
                self.Ha_degraded[ll, ii, :] = np.convolve(self.Ha_I[ll, ii, :],
                                weights, mode = 'same')

    def find_lc_min(self, Plot_B):
        ''' Calculate the line center with non-degraded
        spectral profiles
        '''
        waves = np.linspace(self.wave-.1,self.wave+.1,num=20)
        for xx in range(self.N_nu):
            for yy in range(self.N_phi):
                I = np.interp(waves, self.Ha_waves_u, self.Ha_I[:, xx, yy])
                b = (calc_v_lc(waves, I, 2, -2, 10))
                self.vel_lc[xx, yy] = (b - self.wave) * c /self.wave

                if (Plot_B is True):
                    print(f'b is {b}')

                    plt.plot(self.Ha_waves_u, self.Ha_I[:, xx, yy], 'b.')
                    plt.plot(np.tile(self.vel_lc[xx, yy],
                                     len(self.Ha_waves_u)), self.Ha_I[:, xx, yy], 'r--')
                    plt.xlim(self.wave-.1, self.wave+.1)
                    plt.xlabel('Wavelength, [nm]')
                    plt.show()
        return 0

    def find_lc_min1(self, Plot_B):
        ''' Calculate the line center with the instrument
        degraded spectral profiles
        '''
        waves = np.linspace(self.wave - .1, self.wave + .1, num=40)
        for xx in range(self.N_nu):
            for yy in range(self.N_phi):
                I = np.interp(waves, self.Ha_waves_u, self.Ha_I[:, xx, yy])
                b, coeffs = calc_v_lc(waves, I, 2, 38, 7)
                self.vel_lc[xx, yy] = (b - self.wave) * c /self.wave

                if Plot_B is True:
                    print(f'b is {b}')
                    plt.plot(waves, I, 'b.')
                    plt.plot(waves, parabola(waves, coeffs), 'r.--')
                    plt.xlim(self.wave-.1, self.wave+.1)
                    plt.xlabel('Wavelength, [nm]')
                    plt.show()


    def calc_T(self):
        self.T = np.zeros(self.N_nu)
        for i in range(self.N_nu):
            self.T[i] = np.amax(np.abs(self.vel_lc[i, :])) / self.Amplitude

    def calc_Flux(self):
        self.calc_v_group(self.nu)
        self.Flux = [self.rho_0 * self.Amplitude**2 * self.v_group_c[ii] for
                     ii in range(self.N_nu)]
        # print(f'The calculated flux is {self.Flux}')

    def calc_G(self):
        self.G       = np.zeros(self.N_nu)
        self.var_vel = np.zeros(self.N_nu)
        self.calc_Flux()
        for i in range(self.N_nu):
            # print(f'The flux is {self.Flux[i]} for {np.var(self.vel_lc[i, :])}')
            self.var_vel[i] = np.var(self.vel_lc[i, :])
            self.G[i] =  self.Flux[i] / self.var_vel[i]

    def calc_G_precalc(self):


        G_file = "Ca_8542_vel_lc_G_vel_var.npz"
        with np.load(G_file) as a:
            G_c     = a["Ca_G"]
            vel_rms = a["Ca_var_vel"]
            nu      = a["nu"]


        nu_array = [nu for el in range(vel_rms.shape[0])]
        self.G_approx = sp.interpolate.interp2d(nu_array,
                                                vel_rms.flatten(),
                                                G_c.T.flatten(),
                                                bounds_errors=False,
                                                kind="linear")

    def RF(self):
        '''Calculate the velocity Response function
        from the spectral line
        '''
        self.RF = np.zeros(self.Nx_new_num)
        for ii in range(self.Nx_new_num):
            self.RF[ii] = self.vel_lc[ii, 0]/self.Amplitude

    def v_group(self, nu_i):
        if nu_i > self.nu_ac:
            return self.cs * np.sqrt(1 - (self.nu_ac / nu_i)**2)
        else:
            return 0

    def calc_v_group(self, freqs):
        self.v_group_c = np.zeros(freqs.size)
        for ii in range(freqs.size):
            self.v_group_c[ii] = self.v_group(freqs[ii])

    def calc_T_c(self, freqs):
        self.T_c = np.zeros(freqs.size)
        T = sp.interpolate.interp1d(self.nu, self.T)
        for ii in range(freqs.size):
            self.T_c[ii] = T(freqs[ii])

    def calc_G_c(self, freqs):
        self.G_c = np.zeros(freqs.size)
        G = sp.interpolate.interp1d(self.nu, self.G)
        for ii in range(freqs.size):
            self.G_c[ii] = G(freqs[ii])

    def calc_Acoustic_flux(self, P_i, nu_i, nu_f, slope_index):
        ''' Calc the acoustic flux, given the P_0,
        and the slope of the power law'''

        T = sp.interpolate.interp1d(self.nu, self.T)
        self.rho = 5e-8  # kg / m3
        cs = 7e3  # m/s

        def flux_est(freq):
            return P_i * ((freq/nu_i)
                          ** slope_index) * cs * self.rho/(T(freq) ** 2)

        flux = integrate.quad(flux_est, nu_i, nu_f)
        self.flux = flux
        return flux

    def calc_Acoustic_flux_map(self, freqs, v_rms_map):
        '''


        Parameters
        ----------
        freqs : float array
            Input frequency range in Hertz
        v_rms_map : float array
            Input RMS velocities in km^2/s^2

        Returns
        -------
        flux : float
            Acoustic flux calculated from the input parameters and the G_coeff

        '''
        map_x = v_rms_map.shape[1]
        map_y = v_rms_map.shape[2]

        flux_map = np.zeros((map_x,map_y))
        print(f'Map dimensions are {map_x} and {map_y}')

        self.calc_G_c(freqs)
        dfreq = freqs[1] - freqs[0]
        print(f'G_c is {self.G_c}')
        for ii in range(map_x):
            for jj in range(map_y):
                flux_map[ii, jj] = dfreq * np.sum(self.G_c * v_rms_map[:,ii,jj])
        return flux_map

    def calc_acoustic_flux_analytic(self, freqs, v_rms):
        flux = 0
        self.rho = 5e-8  # kg / m3
        dfreq = freqs[1] - freqs[0]
        for ii in range(freqs.size):
            flux = flux + (dfreq * v_rms[ii] * self.rho
                           *self.v_group_c[ii] / (self.T_c[ii] ** 2))
        return flux

    def calc_Acoustic_flux_map_G_approx(self, freqs, v_rms):
        '''

        Parameters
        ----------
        freqs : float array
            Input frequency range in Hertz
        v_rms : float array
            Input RMS velocities in km^2/s^2

        Returns
        -------
        flux : float

        '''

        G_file = "Ca_8542_vel_lc_G_vel_var.npz"
        with np.load(G_file) as a:
            G_c     = a["Ca_G"]
            vel_rms = a["Ca_var_vel"]
            nu      = a["nu"]


        nu_array = [nu for el in range(vel_rms.shape[0])]
        G_approx = sp.interpolate.interp2d(nu_array,
                                           vel_rms.flatten(),
                                           G_c.flatten(),
                                           kind="linear",
                                           bounds_error=True,)

        map_x = v_rms.shape[1]
        map_y = v_rms.shape[2]
        flux_map = np.zeros((map_x,map_y))
        print(f'Map dimensions are {map_x} and {map_y}')

        self.calc_G_c(freqs)

        dfreq = freqs[1] - freqs[0]
        for ii in range(map_x):
            print(ii)
            for jj in range(map_y):
                for nu in range(len(freqs)):
                    try:
                        G_local = G_approx(freqs[nu], v_rms[nu, ii, jj])
                    except ValueError:
                        if v_rms[nu, ii, jj] > 0:
                            G_local = self.G_c[nu]
                            print(f"{v_rms[nu, ii, jj]:.4f}, {G_local:.2f}")

                    flux_map[ii, jj] = (flux_map[ii, jj]
                                        + dfreq * G_local * v_rms[nu, ii, jj])
        return flux_map



class Instrument_Profile():
    def __init__(self, sampling_rate, SPSF):
        self.sampling_rate = sampling_rate
        self.SPSF          = SPSF


class RADYN_atmos():
    """
    Class for analyzing the outputs from RADYN runs

    iel = 0 is hydrogen
    iel = 1 is ca
    iel = 2 is He

    kr = 2 is H-alpha
    kr = 20 is Ca II 8542

    """
    def __init__(self, cdfFile):
        import radynpy as rp
        self.name        = cdfFile
        self.rad         = rp.cdf.LazyRadynData(cdfFile)
        self.temperature = self.rad.tg1
        self.I           = self.rad.outint
        self.rho         = self.rad.d1
        self.time        = self.rad.time
        self.vz          = self.rad.vz1
        self.pressure    = self.rad.pg1
        self.Ha_n        = 2
        self.Ca_8542     = 6
        self.z           = self.rad.z1
        self.t_steps     = self.z.shape[0]
        self.Nz          = self.z.shape[1]
        self.size        = 191
        self.num_eq      = 4000
        self.Ne          = self.rad.ne1
        self.nH_states   = 6
        self.nH          = self.rad.n1[:, :, :, 0]
        self.vturb       = self.rad.vturb

    def load_tau_scale(self, tau_file):
        '''
        Load the tau scale from the .sav file

        Parameters
        ----------
        tau_file : TYPE
            DESCRIPTION.

        Returns
        -------
        pNone.

        '''

        sav_dic = readsav(tau_file)
        self.tau_scale = sav_dic["tau_all"]
        print(f"Tau scale loaded from file {tau_file}.")

    def interp_atmos_height(self, N_points):
        '''
        Interpolate the atmosphere on a finer shell to remove the shocks

        Parameters
        ----------
        Npoints : int
            Number of points the atmosphere to be interpolated on

        Returns
        -------
        None.

        '''

        new_height = np.linspace(self.z[0, 0], self.z[0, -1], num=N_points)

        self.temperature = np.array([(np.interp((new_height), np.flip(el[0]),
                                                 np.flip(el[1])))
                                     for el in
                                     zip(self.z, self.temperature)])
        self.rho         = np.array([np.interp(new_height, np.flip(el[0]),
                                               np.flip(el[1]))
                                     for el in
                                     zip(self.z, self.rho)])
        self.vz          = np.array([np.interp(new_height, np.flip(el[0]),
                                               np.flip(el[1]))
                                     for el in
                                     zip(self.z, self.vz)])
        self.pressure    = np.array([np.interp(new_height, np.flip(el[0]),
                                               np.flip(el[1]))
                                     for el in
                                     zip(self.z, self.pressure)])

        self.Nz          = N_points
        self.size        = N_points
        self.Ne          = np.array([np.interp(new_height, np.flip(el[0]),
                                               np.flip(el[1])) for
                                     el in zip(self.z, self.Ne)])

        nH1              = np.zeros((self.t_steps, self.Nz, 6))

        for ii in range(self.nH_states):
             nH1[:, :, ii] = np.array([np.interp(new_height, np.flip(el[0]),
                                                 np.flip(el[1]))
                                       for el in
                                       zip(self.z, self.nH[:, :, ii])])

        self.nH = nH1
        self.vturb       = np.interp(new_height, self.z[0], self.vturb)

        self.z = new_height
        print(f"Atmosphere interpolated on {N_points} grid in height.")

    def interp_atmos_cmass(self, N_points):
        '''
        Interpolate the atmosphere on a finer shell to remove the shocks

        Parameters
        ----------
        Npoints : int
            Number of points the atmosphere to be interpolated on

        Returns
        -------
        None.

        '''
        new_height = np.linspace(self.rad.cmass1[0, 0], self.rad.cmass1[0, -1],
                                 num=N_points)

        self.temperature = np.array([(np.interp((new_height), np.flip(el[0]),
                                                 np.flip(el[1])))
                                     for el in
                                     zip(self.cmass1, self.temperature)])
        self.rho         = np.array([np.interp(new_height, np.flip(el[0]),
                                               np.flip(el[1]))
                                     for el in
                                     zip(self.cmass1, self.rho)])
        self.vz          = np.array([np.interp(new_height, np.flip(el[0]),
                                               np.flip(el[1]))
                                     for el in
                                     zip(self.cmass1, self.vz)])
        self.pressure    = np.array([np.interp(new_height, np.flip(el[0]),
                                               np.flip(el[1]))
                                     for el in
                                     zip(self.cmass1, self.pressure)])

        self.Nz          = N_points
        self.size        = N_points
        self.Ne          = np.array([np.interp(new_height, np.flip(el[0]),
                                               np.flip(el[1])) for
                                     el in zip(self.cmass1, self.Ne)])

        nH1              = np.zeros((self.t_steps, self.Nz, 6))

        for ii in range(self.nH_states):
             nH1[:, :, ii] = np.array([np.interp(new_height, np.flip(el[0]),
                                                 np.flip(el[1]))
                                       for el in
                                       zip(self.cmass1, self.nH[:, :, ii])])

        self.nH = nH1
        self.vturb       = np.interp(new_height, self.cmass1[0], self.vturb)

        self.z = new_height
        print(f"Atmosphere interpolated on {N_points} grid in height.")

    def double_num_points(self):
        """

        Double the atmosphere points as plain old average between the points.
        Returns
        -------
        None.

        """



    def cut_coronal_temps(self, T_coronal):
        """
        Remove all the coronal temperatures from the atmosphere going into
        the RH calculation.

        Parameters
        ----------
        T_coronal : float
            Coronal temperatures to clip the atmosphere at

        Returns
        -------
        Atmosphere without a corona.

        """

        idx = (np.abs(self.temperature[-1, :]  - T_coronal)).argmin()


        self.temperature = self.temperature[:, idx:]
        self.rho         = self.rad.d1[:, idx:]
        self.vz          = self.rad.vz1[:, idx:]
        self.pressure    = self.rad.pg1[:, idx:]
        self.Ha_n        = 2
        self.Ca_8542     = 6
        self.z           = self.rad.z1[:, idx:]
        self.Nz          = self.z.shape[1]
        self.size        = self.Nz
        self.Ne          = self.rad.ne1[:, idx:]
        self.nH_states   = 6
        self.nH          = self.rad.n1[:, idx:, :, 0]
        self.vturb       = self.rad.vturb[idx:]



        print(f"The Atmosphere is clipped at {T_coronal:.2e} K at index {idx}.")

    def line_intensity_with_cont(self, kr, muIdx):
        """
        Return the line profile in physical units (cgs) and Å


        From Chris Osborne (Glasgow)
        """

        if not not self.rad.cont[kr]:
            print('line_intensity cannot compute bf intensity')
            return 1

        wl = self.rad.alamb[kr] / (self.rad.q[0:self.rad.nq[kr], kr] *  self.rad.qnorm * 1e5 / self.rad.cc + 1)
        intens = (self.rad.outint[:, 1:self.rad.nq[kr]+1, muIdx, kr]
            + self.rad.outint[:, 0, muIdx, kr][:, np.newaxis]) *  self.rad.cc * 1e8 / (wl**2)[np.newaxis, :]
        return wl[::-1], intens[:,::-1]

    def calc_sp_line(self, kr):
        self.sp_line_wave, self.sp_line_I = self.line_intensity_with_cont(kr, -1)

    def apply_sp_PSF_IBIS(self, num_waves, line):

        if line == "Ca II 8542":
            self.wave_int_grid = np.linspace(self.sp_line_wave[28],
                                             self.sp_line_wave[-28],
                                             num=num_waves)

        elif line == "Ha 6563":
            self.wave_int_grid = np.linspace(self.sp_line_wave[10],
                                             self.sp_line_wave[-10],
                                             num=num_waves)


        line_intensity_int = [np.interp(self.wave_int_grid, self.sp_line_wave, I)
                              for I in self.sp_line_I]

        dfreq = self.wave_int_grid[1] - self.wave_int_grid[0]
        fwhm = 8542 / 1.5e5
        sigma = fwhm / .5 / dfreq
        self.IBIS_sPSF = signal.windows.gaussian(num_waves, sigma)

        sp_line_I_sPSF= [signal.convolve(I, self.IBIS_sPSF, mode="same")
                         for I in line_intensity_int]

        self.line_intensity_int = np.array(sp_line_I_sPSF)

        if line =="Ca II 8542":
            self.IBIS_waves = np.linspace(8544.4-1.05, 8544+1.05, num=41)
        elif line == "Ha 6563":
            self.IBIS_waves = np.linspace(6564.78-1.2, 6564.78+1.2, num=48)
        self.IBIS_int = [np.interp(self.IBIS_waves, self.wave_int_grid, I) for
                         I in self.line_intensity_int]
        self.IBIS_int = np.array(self.IBIS_int)

        #self.sp_line_wave_int for el in :

    def calc_SP_velocity(self):
        self.SP_velocity =  [ 3e5 * ((calc_v_lc(self.sp_line_wave,
                                                self.sp_line_I[i, :], 5, -5, 10)[0])
                                     - 6564.2)/6564.2 for i in range(self.time.size)]
        self.SP_velocity = self.SP_velocity - np.mean(self.SP_velocity)

    def plot_v_field(self, timeStamp):
        plt.figure(figsize=(8,8),dpi=250)
        plt.plot(self.z[:, timeStamp]/1e5, self.vz[:, timeStamp])
        plt.xlabel('Height [km]')
        plt.ylabel('Velocity [cm]')
        plt.show()

    def calc_cs(self, timeStep):
        gamma   = 1.4
        self.cs = [np.sqrt(gamma / self.rho[timeStep, i]
            * self.pressure[timeStep, i]) for i in range(self.size) ]

    def calc_cs_eq(self, timeStep):
        gamma   = 1.4
        self.cs_eq = [np.sqrt(gamma / self.rho_eq[timeStep, i]
            * self.pressure_eq[timeStep, i]) for i in range(self.num_eq)]

    def PSD_velocity(self, height):
        """
        Calculate the PSD of the velocity oscillations at a certain height in
        the atmosphere.

        Parameters
        ----------
        height : float [cm]
            Height of the atmosphere

        Returns
        -------
        self.PSD_vel: array

        """

    def calc_Flux(self,timeStep, dt):
        '''
        Calculate the acoustic flux in a RADYN atmosphere

        Parameters
        ----------
        timeStep : int
            Around which timestep to evaluate the acoustic flux
        dt : int
            interval around which to average to find the flux

        Returns
        -------
        self.Flux -- array containing the calculated flux

        '''
        self.calc_cs(timeStep)
        self.Flux = [np.std(self.vz[(timeStep-dt):(timeStep+dt),i])**2
                     * self.cs[i] * self.rho[timeStep, i]
                     for i in range(self.size)]
        self.Flux = np.array(self.Flux)

    def calc_Flux_equidistant(self, timeStep, dt):
        """
        Calculate the Acoustic flux around time <Timestep> +- dt

        Parameters
        ----------
        timeStep : TYPE
            center of the timestep the flux to be calculated around

        dt : TYPE
            time range around the time step the flux to be calculated around

        Returns
        -------
        None.

        """
        self.z_eq = np.linspace(self.z[0, -1], self.z[0, 0],
                                num=self.num_eq)

        self.pressure_eq = np.zeros((self.t_steps, self.num_eq))
        self.rho_eq      = np.zeros((self.t_steps, self.num_eq))
        self.v_eq        = np.zeros((self.t_steps, self.num_eq))

        for ii in range(self.t_steps):
            self.pressure_eq[ii, :] = np.interp(self.z_eq,
                    np.flip(self.z[ii, :]), np.flip(self.pressure[ii, :]))
            self.rho_eq[ii, :] = np.interp(self.z_eq,
                    np.flip(self.z[ii, :]), np.flip(self.rho[ii, :]))
            self.v_eq[ii, :] = np.interp(self.z_eq,
                    np.flip(self.z[ii, :]), np.flip(self.vz[ii, :]))


        self.calc_cs_eq(timeStep)
        self.Flux_eq = [np.std(self.v_eq[(timeStep-dt):(timeStep+dt),
            i]/1e2)**2 * self.cs_eq[i]/1e2 * self.rho_eq[timeStep, i]*1e3
            for i in range(self.num_eq)]
        self.Flux_eq = np.array(self.Flux_eq)

    def calc_Flux_PSD(self, timeStep, dt, height, dt_sim):
        """
        Calculate the Acoustic flux around time <Timestep> +- dt

        Parameters
        ----------
        timeStep : float [num_tSteps]
            center of the timestep the flux to be calculated around.

        dt : float [num_tSteps]
            time range around the time step the flux to be calculated around.

        height: float [km]
            Height [km] at which the flux to be estimated.

        dt_sim: float [seconds]
            Time step between simulation snapshots analysed.
        Returns
        -------
        self.Flux_eq_PSD_freq -- the frequencies array for the PSD
        self.Flux_eq_PSD -- the Acoustic flux PSD

        """
        self.z_eq = np.linspace(self.z[0, -1], self.z[0, 0],
                                num=self.num_eq)

        self.pressure_eq = np.zeros((self.t_steps, self.num_eq))
        self.rho_eq      = np.zeros((self.t_steps, self.num_eq))
        self.v_eq        = np.zeros((self.t_steps, self.num_eq))

        for ii in range(self.t_steps):
            self.pressure_eq[ii, :] = np.interp(self.z_eq,
                    np.flip(self.z[ii, :]), np.flip(self.pressure[ii, :]))
            self.rho_eq[ii, :] = np.interp(self.z_eq,
                    np.flip(self.z[ii, :]), np.flip(self.rho[ii, :]))
            self.v_eq[ii, :] = np.interp(self.z_eq,
                    np.flip(self.z[ii, :]), np.flip(self.vz[ii, :]))


        self.calc_cs_eq(timeStep)

        idx = (np.abs(self.z_eq/1e5 - height)).argmin()
        print(idx)

        v_freq, v_PSD = signal.periodogram(self.v_eq[(timeStep-dt):(timeStep+dt),
                                                     idx]/1e2,
                                           fs=1/dt_sim)
        self.Flux_eq_PSD_freq = v_freq
        self.Flux_eq_PSD = (v_PSD * (np.mean(self.rho_eq[:, idx]) * 1e3)
                            * (self.cs_eq[idx] / 1e2))


    def calc_tau_surface(self, tau_level):
        """
        Calculate the iso-tau surface in a simulation and return array of
        the heights of iso-tau surface.

        Parameters
        ----------
        tau_level : int
            tau_level to be pursuied

        Returns
        -------
        self.tau_surface : height of tau surface in the simulation;

        """

        self.tau_height     = np.zeros(self.t_steps)
        self.tau_height_idx = np.zeros(self.t_steps)

        for t_step in range(self.t_steps):
            max_tau_idx = self.tau_scale[t_step, -4, :].argmax()
            idx = (np.abs(self.tau_scale[t_step, :, max_tau_idx]
                                - tau_level)).argmin()
            self.tau_height_idx[t_step] = idx
            self.tau_height[t_step] = self.z[t_step, idx]

    def include_V_pulse(self, index, Amplitude):
        self.V = np.zeros(self.Nx_num_new)
        self.V[index] = Amplitude

    def write_RF_v_test(self, t_step):
        ''' Write an RH output atmosphere for Ricardos's version
        of RH from a RADYN computation.
        Format has to be the following:
        Height,Temperature,Ne,V,Vturb,field,gamma,phi,
        nh(1),nh(2),nh(3),nh(4),nh(5),np
        '''

        B           = np.zeros(self.Nz)
        phi         = np.zeros(self.Nz)
        gamma       = np.zeros(self.Nz)

        atmos = np.zeros((self.Nz, 1, 14, self.Nz))



        for ii in range(int(self.Nz-2)):
            try:
                atmos[ii, 0, 0, :] = (self.z[t_step, :])/1e5
            except IndexError:
                atmos[ii, 0, 0, :] = self.z/1e5
            atmos[ii, 0, 1, :] = (self.temperature[t_step, :])
            atmos[ii, 0, 2, :] = (self.rad.ne1[t_step, :])
            atmos[ii, 0, 3, (ii-1):(ii+1)] = 1.0
            atmos[ii, 0, 4, :] = (self.vturb)/1e5
            atmos[ii, 0, 5, :] = (B)
            atmos[ii, 0, 6, :] = (phi)
            atmos[ii, 0, 7, :] = (gamma)

            #rad.nk1[0] is the # of levels in H


            for kk in range(int(self.rad.nk[0])):
                atmos[ii, 0, (8+kk), :] = (self.rad.n1[t_step, :, kk,
                                                       0])

        atmos = np.swapaxes(atmos, 1, 2)
        atmos = np.swapaxes(atmos, 0, 3)

        hdu = fits.PrimaryHDU(atmos)
        hdu.writeto(self.name + "_RF_test.fits", overwrite=True)

        print('Successfully written out atmospher for RF_v test!')


    def write_RF_T_test(self, t_step):
        ''' Write an RH output atmosphere for Ricardos's version
        of RH from a RADYN computation.
        Format has to be the following:
        Height,Temperature,Ne,V,Vturb,field,gamma,phi,
        nh(1),nh(2),nh(3),nh(4),nh(5),np
        '''

        B           = np.zeros(self.Nz)
        phi         = np.zeros(self.Nz)
        gamma       = np.zeros(self.Nz)

        atmos = np.zeros((self.Nz, 1, 14, self.Nz))



        for ii in range(int(self.Nz-2)):
            try:
                atmos[ii, 0, 0, :] = (self.z[t_step, :])/1e5
            except IndexError:
                atmos[ii, 0, 0, :] = self.z/1e5
            atmos[ii, 0, 1, :] = (self.temperature[t_step, :]+100)
            atmos[ii, 0, 2, :] = (self.rad.ne1[t_step, :])
            atmos[ii, 0, 3, :] = (self.vz[t_step, :])/1e5
            atmos[ii, 0, 4, :] = (self.vturb)/1e5
            atmos[ii, 0, 5, :] = (B)
            atmos[ii, 0, 6, :] = (phi)
            atmos[ii, 0, 7, :] = (gamma)

            #rad.nk1[0] is the # of levels in H


            for kk in range(int(self.rad.nk[0])):
                atmos[ii, 0, (8+kk), :] = (self.rad.n1[t_step, :, kk,
                                                       0])

        atmos = np.swapaxes(atmos, 1, 2)
        atmos = np.swapaxes(atmos, 0, 3)

        hdu = fits.PrimaryHDU(atmos)
        hdu.writeto(self.name + "_RF_test.fits", overwrite=True)

        print('Successfully written out atmospher for RF_v test!')

    def write_TF_test(self, write_Ne):
        ''' Write an RH output atmosphere for Ricardos's version
        of RH from a RADYN computation.
        Format has to be the following:
        Height,Temperature,Ne,V,Vturb,field,gamma,phi,
        nh(1),nh(2),nh(3),nh(4),nh(5),np
        '''

        B           = np.zeros(self.Nz)
        phi         = np.zeros(self.Nz)
        gamma       = np.zeros(self.Nz)

        atmos = np.zeros((self.t_steps, 1, 14, self.Nz))



        for ii in range(self.t_steps):
            try:
                atmos[ii, 0, 0, :] = (self.z[ii, :])/1e5
            except IndexError:
                atmos[ii, 0, 0, :] = self.z/1e5
            atmos[ii, 0, 1, :] = (self.temperature[ii, :])
            atmos[ii, 0, 2, :] = (self.Ne[ii, :])
            atmos[ii, 0, 3, :] = self.vz[ii, :]/1e5
            atmos[ii, 0, 4, :] = (self.vturb)/1e5
            atmos[ii, 0, 5, :] = (B)
            atmos[ii, 0, 6, :] = (phi)
            atmos[ii, 0, 7, :] = (gamma)

            #rad.nk1[0] is the # of levels in H

            if write_Ne is True:
                for kk in range(int(self.rad.nk[0])):
                    atmos[ii, 0, (8+kk), :] = (self.nH[ii, :, kk])

            if write_Ne is False:
                for kk in range(int(self.rad.nk[0])):
                    atmos[ii, 0, (8+kk), :] = 0



        atmos = np.swapaxes(atmos, 1, 2)
        atmos = np.swapaxes(atmos, 0, 3)

        hdu = fits.PrimaryHDU(atmos)
        hdu.writeto(self.name + '.fits', overwrite=True)

        print('Successfully written out atmosphere')

def write_rh_output(Atmos_Object, nu_range, Amplitude, N_trials):
    ''' Write an RH output atmosphere for Ricardos's version
        of RH
        Format has to be the following:
        Height,Temperature,Ne,V,Vturb,field,gamma,phi,nh(1),nh(2),nh(3),nh(4),nh(5),np
    '''

    atmos = np.zeros((N_trials, 1, 14, Atmos_Object.Nx_new_num))

    for i in range(N_trials):
        atmos[i, 0, 0, :] = Atmos_Object.height
        atmos[i, 0, 1, :] = Atmos_Object.T
        atmos[i, 0, 2, :] = Atmos_Object.Ne
        atmos[i, 0, 4, :] = Atmos_Object.Vturb
        atmos[i, 0, 5, :] = Atmos_Object.B
        atmos[i, 0, 6, :] = Atmos_Object.phi
        atmos[i, 0, 7, :] = Atmos_Object.gamma
        atmos[i, 0, 8:13, :] = Atmos_Object.nH[0, :]
        Atmos_Object.include_V_perturb(nu, np.random.random_sample() * 2
                                       * np.pi, Amplitude)
        atmos[i, 0, 3, :] = Atmos_Object.V

    atmos = np.swapaxes(atmos, 1, 2)
    atmos = np.swapaxes(atmos, 0, 3)
    hdu = fits.PrimaryHDU(atmos)
    hdu.writeto('/Users/molnarad/Desktop/rh/FAL11_nu_'
                + str(nu) + '_' + str(N_trials) + '.fits', overwrite=True)

    return 'Successfully written out atmosphere'
