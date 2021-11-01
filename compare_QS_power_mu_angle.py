#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:28:02 2021

Compare the QS velocity fields at different mu angles
from the 20130416 rasters

@author: molnarad
"""


import numpy as np
from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D
from matplotlib import pyplot as pl


file_names = ["../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.2.sav",
              "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.4.sav",
              "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.6.sav",
              "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=0.8.sav",
              "../IRIS_data/QS_data/iris_l2_201311167_QS_SitAndStare_mu=1.0.sav",
              #"../IRIS_data/QS_data/iris_l2_20131116_073345_3803010103_raster_t000_r00000_mg2_vel_mu=1.0.sav",
              "../IRIS_data/QS_data/iris_l2_20140201_104147_4004257547_raster_t000_r00000_mu06_mg2_vel.sav"]


d= "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Meetings/Meetings_with_Steve/2021_02_25/"

cadence = [16.7, 16.7, 16.7, 16.7, 16.7, 16.7, 16.7, 16.7, 5]
av_ind = [[100, 250], [400, 450], [600, 710], [480, 650], [20,400], [0, 40]]

#Do the 2v lines first 

dd_param = 1
slit_av_param = 1


asp= "auto" # 0.075/dd_param/2
asp_pxx = "auto" # 5/dd_param*5

t_limits = [1, -2]
slit_limits = [10,-10]

# bp/lc/rp = [Nt, Nslit, v[0]/I[1], line h[0]/k[1]]

line_index = 1

pl.figure(dpi=250)

for el in range(6):
    file_index = el
    file_name = file_names[file_index]
    mu_angle = file_name[-7:-4]
    print(f"The mu angle is {mu_angle}")
    dt      = cadence[file_index]
    bp, rp, lc = load_velocity_mg2(file_name)

    k_2v_v = filter_velocity_field(bp[t_limits[0]:t_limits[1], 
                                   slit_limits[0]:slit_limits[1], 0, line_index], 
                                   dt=7, dd=5, degree=2)


    # plot_velocity_map(k_2v_v, title="QS k2v velocity @ mu="+mu_angle, 
    #                   aspect=asp, d=d, cadence = dt)
    freq, Pxx_k_2v_v = calc_Pxx_velmap(k_2v_v, fsi=1/dt)

    
    Pxx_average = np.nanquantile(Pxx_k_2v_v[av_ind[el][0]:av_ind[el][1], :],
                                 0.5, axis=0)
    
    pl.loglog(freq, Pxx_average/1e3, '.--', label="$\\mu$ = " + mu_angle)
    print(f"Total power between 5 and 20 mHz is: " + 
          f"{freq[1]*np.sum(Pxx_average[10:40]):.1f}")
    
pl.grid(alpha=0.25)
pl.legend(fontsize=8)
pl.xlabel("Frequency [mHz]")
pl.ylabel("Power [(km/s)$^2$/mHz]")
pl.ylim(1e-2, 3e0)
pl.title("Average PSD from QS 20131116")
pl.show()