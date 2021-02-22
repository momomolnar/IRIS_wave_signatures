#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 22:50:37 2021
Make SJI Movie from the IRIS SJI dataset from 2017 23 April
@author: molnarad
"""

import numpy as np
from matplotlib import pyplot as pl
from scipy.io import readsav
from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D



data_dir = ("/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves"
            + "/IRIS_data/Apr_2017_data/")

file_name = "IRIS_SJI_raster_aligned_IBIS.sav"

sav_data = readsav(data_dir + file_name)

iris_data = sav_data["iris_SJI_fin"]

for el in range(0, 1):
    pl.figure(dpi=250)
    pl.imshow(iris_data[el, :, :], vmax=80, vmin=2, origin="lower",
              extent=[-127.98, -31.98, 210.99, 306.99])
    time = int(6.32*8*el)
    pl.title(f"Frame {el:{0}{3}} / {time:{0}{5}} sec")
    pl.tight_layout()
    pl.savefig((data_dir+"SJI_movie_frames/frame_" + "{0:0=3d}".format(el)
                + ".jpg"), transparent=True)
    pl.ylabel("Solar Y [arcsec]")
    pl.xlabel("Solar X [arcsec]")
    pl.show()

