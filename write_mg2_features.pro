; Script to write out the Mg II h& K properties derived with the 
; iris_get_mg_features_lev2

data_dir = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Plage_data/"
files = find_file(data_dir + "*.fits", count=n_files)
vr = [-30, 30]

; mg2_index = [7, 6, 7, 7, 7, 6] ; CH data
mg2_index = [7, 2, 7, 7, 7, 7, 2, 2, 7]
mu_angle  = ["_0.44", "_0.67", "_0.87", "_0.48", "_1.00", "_1.00", "_0.39", "_0.22", "_0.70"]

FOR el=4, (n_files-1) DO BEGIN
  print, el, " ", mu_angle[el], " ",  files[el] 
  iris_get_mg_features_lev2, files[el], mg2_index[el], vr, lc, rp, bp
  file_name_out = files[el].Remove(-5) + "_" + mu_angle[el] + "_mg2_vel.sav"
  save, lc, rp, bp, filename=file_name_out
ENDFOR

END
