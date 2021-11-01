data_dir = "../IRIS_data/Apr_2017_data/iris_l2_20170423_152532_3624505432_raster/" 

files = find_files("*.fits", data_dir) 

num_exten = 8 ; 7 Mg h/k lines

start_index = 2 
end_index = 499

num_files = end_index - start_index

example_file = readfits(files[0], exten=num_exten, hdr_aa)
size_file    = size(example_file)
num_waves    = size_file[1]
num_ypix     = size_file[2]
num_steps    = size_file[3]
 
dy = 12 ; number of IRIS pixels (0.166 arcsec) per one step of the IRIS raster 
        ; which is 1.996 arcseconds

data_cube_bp_v = fltarr(1000, 1000, 2, num_files)
data_cube_rp_v = fltarr(1000, 1000, 2, num_files)
data_cube_lc_v = fltarr(1000, 1000, 2, num_files)
data_cube_bp_I = fltarr(1000, 1000, 2, num_files)
data_cube_rp_I = fltarr(1000, 1000, 2, num_files)
data_cube_lc_I = fltarr(1000, 1000, 2, num_files)

temp_empty_bp = fltarr(978, 978, 2, 2)
temp_empty_lc = fltarr(978, 978, 2, 2)
temp_empty_rp = fltarr(978, 978, 2, 2)
 
vl = [-40, 40]
FOR ii= 0, (num_files-1) DO BEGIN
    ; temp_file = readfits(files[ii], exten = num_exten)
    iris_get_mg_features_lev2, files[ii], 7, vl, lc, bp, rp
    FOR step=0, (num_steps-1) DO BEGIN
        FOR jj=0, (dy-1) DO BEGIN
            dx0 = 418 + step*dy + jj
            temp_empty_bp[dx0, 343:741, *, *] = $
                transpose(bp[*, *, *, step], [2, 0, 1])
            temp_empty_lc[dx0, 343:741, *, *] = $
                transpose(lc[*, *, *, step], [2, 0, 1])
            temp_empty_rp[dx0, 343:741, *, *] = $
                transpose(rp[*, *, *, step], [2, 0, 1])
            print, "step is", ii, step, jj, dx0
        ENDFOR
    ENDFOR

    empty1 = temp_empty_rp[220:798, 220:798, *, 0]
    empty_IBIS = CONGRID(empty1, 1000, 1000, 2) 
    data_cube_rp_v[*, *, *,  ii] = empty_IBIS
    empty1 = temp_empty_bp[220:798, 220:798, *, 0]
    empty_IBIS = CONGRID(empty1, 1000, 1000, 2) 
    data_cube_bp_v[*, *, *, ii] = empty_IBIS
    empty1 = temp_empty_lc[220:798, 220:798, *, 0]
    empty_IBIS = CONGRID(empty1, 1000, 1000, 2) 
    data_cube_lc_v[*, *, *, ii] = empty_IBIS
    empty1 = temp_empty_rp[220:798, 220:798, *, 1]
    empty_IBIS = CONGRID(empty1, 1000, 1000, 2) 
    data_cube_rp_I[*, *, *, ii] = empty_IBIS
    empty1 = temp_empty_bp[220:798, 220:798, *, 1]
    empty_IBIS = CONGRID(empty1, 1000, 1000, 2) 
    data_cube_bp_I[*, *, *, ii] = empty_IBIS
    empty1 = temp_empty_lc[220:798, 220:798, *, 0]
    empty_IBIS = CONGRID(empty1, 1000, 1000, 2) 
    data_cube_lc_I[*, *, *, ii] = empty_IBIS

ENDFOR

save, data_cube_bp_I, data_cube_rp_I, $
      data_cube_lc_I, hdr_aa, $
      data_cube_bp_v, data_cube_rp_v, data_cube_lc_v,$
      filename="IRIS_raster_IBIS_aligned.2017Apr23.sav" 
END
