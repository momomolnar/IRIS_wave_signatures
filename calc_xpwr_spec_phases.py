#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:52:17 2021
Calculate some cross-wavelet-spectra, coherencies and their phases;
@author: molnarad
"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.image import NonUniformImage

from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D

import pycwt as wavelet
from pycwt.helpers import find



file_name = ["../IRIS_data/CH_data/CH_vel_mgk_mu=0.99.sav",
             "../IRIS_data/CH_data/CH_vel_mgk_mu=0.93.sav",
             "../IRIS_data/CH_data/CH_vel_mgk_mu=0.83.sav",
             "../IRIS_data/CH_data/CH_vel_mgk_mu=0.75.sav",
             "../IRIS_data/CH_data/CH_vel_mgk_mu=0.65.sav",
             "../IRIS_data/CH_data/CH_vel_mgk_mu=0.34.sav",]

file_index = 2
file_name = file_name[file_index]
mu_angle = file_name[-8:-4]
print(f"The mu angle is {mu_angle}")
d= ("/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Meetings/"
    + "Meetings_with_Steve/2021_01_14/")


cadence = [9.47, 16.39, 9.33, 9.33, 16.21, 9.57]
dt      = cadence[file_index]
bp, rp, lc = load_velocity_mg2(file_name)

dd_param = 1
slit_av_param = 1


asp= 0.075/dd_param*1
asp_pxx = 5/dd_param*2.5

t_limits = [1, -5]
slit_limits = [10,-100]

k_2v_v = filter_velocity_field(bp[t_limits[0]:t_limits[1], 
                                  slit_limits[0]:slit_limits[1], 0, 0], 
                               dd=dd_param)
plot_velocity_map(k_2v_v, title="CH k2v velocity @ mu="+mu_angle, 
                  aspect=asp, d=d, cadence = dt)


k_3_v = filter_velocity_field(lc[t_limits[0]:t_limits[1], 
                                 slit_limits[0]:slit_limits[1], 0, 0], 
                              dd=dd_param)
plot_velocity_map(k_3_v, title="CH k3 velocity @ mu="+mu_angle, 
                  aspect=asp, d=d, cadence=dt)

title = 'IRIS Mg II k2v Velocity'
label = 'k2v-k3 Vel'
units = 'km s$^{-1}$'
t0 = 0
dt = dt  # In seconds

N = k_2v_v.shape[1]
t = np.arange(0, N) * dt + t0

k2v_ex = k_2v_v[225, :]
std_k2v = k2v_ex.std()  # Standard deviation
var_k2v = std_k2v ** 2  # Variance
k2v_norm = k2v_ex / std_k2v  # Normalized dataset


k3v_ex = k_3_v[225, :]
std_k3v = k3v_ex.std()  # Standard deviation
var_k3v = std_k3v ** 2  # Variance
k3v_norm = k3v_ex / std_k3v  # Normalized dataset

t1 = t2 = np.linspace(0, k3v_ex.size - 1, num=k3v_ex.size)*dt

mother = wavelet.Morlet(6)
s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 secondss = 6 months
dj = 1 / 12  # Twelve sub-octaves per octaves
J = 7 / dj  # Seven powers of two with dj sub-octaves
alpha, _, _ = wavelet.ar1(k2v_ex)  # Lag-1 autocorrelation for red noise

d = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/QS_data/"
filename = "iris_l2_20131116_073345_3803010103_raster_t000_r00000MnI_2801.9.sav"

aa = io.readsav(d+filename)

fit_params = aa["fit_params"]

velocities = fit_params[1, :, :]

rest_wave = np.median(velocities) + 2801.95

velocities = (velocities - rest_wave)*3e5/rest_wave

vel_reduced_Mn = filter_velocity_field(velocities.T,  
                                    dv=1, dd=3, degree=1)

plot_velocity_map(vel_reduced, vmin=-2, vmax=2, aspect=0.1, 
                  title="Mn I 2801.5 $\AA$ Doppler velocity ")


wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(k2v_ex, dt, dj, s0, J,
                                                      mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std_k2v

power = (np.abs(wave)) ** 2
fft_power = np.abs(fft) ** 2
period = 1 / freqs

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig95

glbl_power = power.mean(axis=1)
dof = N - scales  # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(var_k2v, dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)


# Prepare the figure
pl.close('all')
pl.ioff()
figprops = dict(figsize=(11, 8), dpi=144)
fig = pl.figure(**figprops)

# First sub-plot, the original time series anomaly and inverse wavelet
# transform.
ax = pl.axes([0.1, 0.75, 0.65, 0.2])
ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(t, k2v_ex, 'k', linewidth=1.5)
ax.set_title('a) {}'.format(title))
ax.set_ylabel(r'{} [{}]'.format(label, units))

# Second sub-plot, the normalized wavelet power spectrum and significance
# level contour lines and cone of influece hatched area. Note that period
# scale is logarithmic.
bx = pl.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
            extend='both', cmap=pl.cm.viridis)
extent = [t.min(), t.max(), 0, max(period)]
bx.contour(t, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
           extent=extent)
bx.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                           t[:1] - dt, t[:1] - dt]),
        np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                           np.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(label, mother.name))
bx.set_ylabel('Period (seconds)')
#
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                           np.ceil(np.log2(period.max())))
bx.set_yticks(np.log2(Yticks))
bx.set_yticklabels(Yticks)
bx.set_xlabel("Time [seconds]")

# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra. Note that period scale is logarithmic.
cx = pl.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, np.log2(period), 'k--')
cx.plot(var_k2v * fft_theor, np.log2(period), '--', color='#cccccc')
cx.plot(var_k2v * fft_power, np.log2(1./fftfreqs), '-', color='#cccccc',
        linewidth=1.)
cx.plot(var_k2v * glbl_power, np.log2(period), 'k-', linewidth=1.5)
cx.set_title('c) Global Wavelet Spectrum')
cx.set_xlabel(r'Power [({})$^2$]'.format(units))
cx.set_xlim([0, (glbl_power * var_k2v).max()])
cx.set_ylim(np.log2([period.min(), period.max()]))
cx.set_yticks(np.log2(Yticks))
cx.set_yticklabels(Yticks)
#cx.set_xlabel('Time (seconds)')
pl.setp(cx.get_yticklabels(), visible=False)
pl.savefig(d+"wavelet_test_k2v.png")
pl.show()


W12, cross_coi, freq, signif = wavelet.xwt(k2v_ex, k3v_ex, dt, dj=1/12, s0=s0,
                                           J=-1,
                                           significance_level=0.8646,
                                           wavelet='morlet', normalize=True)

cross_power = np.abs(W12)**2
cross_sig = np.ones([1, k2v_ex.size]) * signif[:, None]
cross_sig = cross_power / cross_sig  # Power is significant where ratio > 1
cross_period = 1/freq

# Calculate the wavelet coherence (WTC). The WTC finds regions in time
# frequency space where the two time seris co-vary, but do not necessarily have
# high power.
WCT, aWCT, corr_coi, freq, sig = wavelet.wct(k2v_ex, k3v_ex, dt, dj=1/12, 
                                             s0=s0, J=-1,
                                             significance_level=0.8646,
                                             wavelet='morlet', normalize=True,
                                             cache=True)

cor_sig = np.ones([1, k2v_ex.size]) * sig[:, None]
cor_sig = np.abs(WCT) / cor_sig  # Power is significant where ratio > 1
cor_period = 1 / freq

# Calculates the phase between both time series. The phase arrows in the
# cross wavelet power spectrum rotate clockwise with 'north' origin.
# The relative phase relationship convention is the same as adopted
# by Torrence and Webster (1999), where in phase signals point
# upwards (N), anti-phase signals point downwards (S). If X leads Y,
# arrows point to the right (E) and if X lags Y, arrow points to the
# left (W).
angle = 0.5 * np.pi - aWCT
u, v = np.cos(angle), np.sin(angle)

plot_every = 5

fig, (ax1, ax2) = pl.subplots(nrows=2, ncols=1, sharex=True, 
                              sharey=True, dpi=250, figsize=(7.5, 10))
fig.subplots_adjust(right=0.8)

levels = np.linspace(np.log10(np.amin(cross_power)), np.log10(np.amax(cross_power)), num=7)
im1 = ax1.contourf(t, np.log2(cross_period), np.log10(cross_power), (levels),
                   extend='both', cmap=pl.cm.viridis)
extent = [t.min(), t.max(), 0, max(period)]
ax1.contour(t1, np.log2(cross_period), cross_sig, [-99, 1], colors='k', linewidths=2,
           extent=extent)
ax1.fill(np.concatenate([t1, t1[-1:] + dt, t1[-1:] + dt,
                        t1[:1] - dt, t1[:1] - dt]),
        np.concatenate([np.log2(cross_coi), [1e-9], np.log2(cross_period[-1:]),
                        np.log2(cross_period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
ax1.set_title('a) {} X Wavelet Power Spectrum ({})'.format(label, mother.name))
ax1.set_ylabel('Period (seconds)')
ax1.quiver(t1[::plot_every], np.log2(cross_period[::plot_every]), 
           u[::plot_every, ::plot_every], 
           v[::plot_every, ::plot_every],
           units='width', angles='uv', pivot='mid', linewidth=1,
           edgecolor='k', headwidth=10, headlength=10, headaxislength=5,
           minshaft=2, minlength=5)

Yticks = 2 ** np.arange(np.ceil(np.log2(cross_period.min())),
                        np.ceil(np.log2(cross_period.max())))
ax1.set_ylim(np.log2(32),  np.log2(750))
ax1.set_yticks(np.log2(Yticks))
ax1.set_yticklabels(Yticks)

ax2.set_xlabel("Time [seconds]")
#fig.colorbar(im2, cax=cbar_ax_1)
fig.colorbar(im1, ax=ax1, label="X wavelet power")
"""
cbar_ax = fig.add_axes([0.85, 0.55, 0.05, 0.35])
cbar_ax_1 = fig.add_axes([0.85, 0.05, 0.05, 0.35])

extent_cross = [t1.min(), t1.max(), 0, max(cross_period)]
extent_corr = [t1.min(), t1.max(), 0, max(cor_period)]
im1 = NonUniformImage(ax1, interpolation='bilinear', extent=extent_cross)
im1.set_data(t1, cross_period, cross_power)
ax1.images.append(im1)
ax1.contour(t1, cross_period, cross_sig, [-99, 1], colors='k', linewidths=2,
            extent=extent_cross)
ax1.fill(np.concatenate([t1, t1[-1:]+dt, t1[-1:]+dt, t1[:1]-dt, t1[:1]-dt]),
         np.concatenate([cross_coi, [1e-9], cross_period[-1:],
                         cross_period[-1:], [1e-9]]),
         'k', alpha=0.3, hatch='x')
ax1.set_title('Cross-Wavelet')
ax1.quiver(t1[::plot_every], cross_period[::plot_every], 
           u[::plot_every, ::plot_every], 
           v[::plot_every, ::plot_every],
           units='width', angles='uv', pivot='mid', linewidth=1,
           edgecolor='k', headwidth=10, headlength=10, headaxislength=5,
           minshaft=2, minlength=5)
ax1.set_ylabel("Period [s]")
fig.colorbar(im1, cax=cbar_ax)
"""
'''
im2 = NonUniformImage(ax2, interpolation='bilinear', extent=extent_corr)
im2.set_data(t1, cor_period, WCT)
ax2.images.append(im2)
ax2.contour(t1, cor_period, cor_sig, [-99, 1], colors='k', linewidths=2,
            extent=extent_corr)
ax2.fill(np.concatenate([t1, t1[-1:]+dt, t1[-1:]+dt, t1[:1]-dt, t1[:1]-dt]),
         np.concatenate([corr_coi, [1e-9], cor_period[-1:], cor_period[-1:],
                         [1e-9]]),
         'k', alpha=0.3, hatch='x')
ax2.set_title('Cross-Correlation')
ax2.set_ylabel("Period [s]")
ax2.set_xlabel("Time [s]")
ax2.quiver(t1[::plot_every], cross_period[::plot_every], 
           u[::plot_every, ::plot_every], 
           v[::plot_every, ::plot_every],
           units='height',
           angles='uv', pivot='mid', linewidth=1, edgecolor='k',
           headwidth=10, headlength=10, headaxislength=5, minshaft=2,
           minlength=5)
#ax2.set_ylim(2, 35)
#ax2.set_xlim(max(t1.min(), t2.min()), min(t1.max(), t2.max()))
fig.colorbar(im2, ax=ax2, cax=cbar_ax_1)
'''
# Second sub-plot, the normalized wavelet power spectrum and significance
# level contour lines and cone of influece hatched area. Note that period
# scale is logarithmic.
levels = np.linspace(0, 1, num=10)
im2 = ax2.contourf(t, np.log2(cor_period), WCT, (levels),
                   extend='both', cmap=pl.cm.viridis)
extent = [t.min(), t.max(), 0, max(period)]
ax2.contour(t, np.log2(cor_period), cor_sig, [-99, 1], colors='k', linewidths=2,
           extent=extent)
ax2.fill(np.concatenate([t1, t1[-1:] + dt, t1[-1:] + dt,
                        t1[:1] - dt, t1[:1] - dt]),
        np.concatenate([np.log2(corr_coi), [1e-9], np.log2(cor_period[-1:]),
                        np.log2(cor_period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
ax2.set_title('b) {} X Wavelet Coherence Spectrum ({})'.format(label, mother.name))
ax2.set_ylabel('Period (seconds)')
ax2.quiver(t1[::plot_every], np.log2(cor_period[::plot_every]), 
           u[::plot_every, ::plot_every], 
           v[::plot_every, ::plot_every],
           units='height',
           angles='uv', pivot='mid', linewidth=1, edgecolor='k',
           headwidth=10, headlength=10, headaxislength=5, minshaft=2,
           minlength=5)

Yticks = 2 ** np.arange(np.ceil(np.log2(cor_period.min())),
                        np.ceil(np.log2(cor_period.max())))
ax2.set_ylim(np.log2(32),  np.log2(750))
ax2.set_yticks(np.log2(Yticks))
ax2.set_yticklabels(Yticks)

ax2.set_xlabel("Time [seconds]")
#fig.colorbar(im2, cax=cbar_ax_1)
fig.colorbar(im2, ax=ax2, label="Coherency")

pl.draw()
pl.show()

pl.draw()
pl.show()