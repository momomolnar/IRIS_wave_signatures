#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 23:07:00 2021

Compare the IRIS and IBIS V_rms for the different 


@author: molnarad
"""


import numpy as np
from matplotlib import pyplot as pl
from scipy.io import readsav
from IRIS_lib import filter_velocity_field, load_velocity_mg2, plot_velocity_map
from IRIS_lib import plot_intensity_map, average_velmap_slitwise
from IRIS_lib import calc_Pxx_velmap, plot_Pxx_2D

