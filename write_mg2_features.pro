file_name_in = "../IRIS_data/CH_data/iris_l2_20131116_073345_3803010103_raster_t000_r00000.fits"

vr = [-40, 40] ; Velocity Range about line center to search for features
iris_get_mg_features_lev2, file_name_in, 6, vr, lc, rp, bp

file_name_out = file_name_in.Remove(-5) + "_mg2_vel.sav"

save, lc, rp, bp, filename=file_name_out

END
