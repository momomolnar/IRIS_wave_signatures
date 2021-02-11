
data_dir = "../IRIS_data/Apr_2017_data/iris_l2_20170423_152532_3624505432_raster/" 

files = find_files("*.fits", data_dir) 

num_exten = 8 ; 7 Mg h/k lines

start_index = 240 
end_index = 242

num_files = end_index - start_index

example_file = readfits(files[0], exten=num_exten, hdr_aa)
size_file    = size(example_file)
num_waves    = size_file[1]
num_ypix     = size_file[2]
num_steps    = size_file[3]

dy = 12

data_cube = fltarr(1000, 1000, num_waves, num_files)

temp_empty = fltarr(978, num_waves, 978)
 
FOR ii= 0, (num_files-1) DO BEGIN
    temp_file = readfits(files[ii], exten = num_exten)
    
    FOR step=0, (num_steps-1) DO BEGIN
        FOR jj=0, (dy-1) DO BEGIN
            dx0 = 418 + step*12 + jj
            temp_empty[dx0, *, 343:741] = $
                temp_file[*, *, step]
        ENDFOR
    ENDFOR

    empty1 = temp_empty[220:798, *, 220:798]
    empty_IBIS = CONGRID(empty1, 1000, num_waves, 1000)
    empty_IBIS = transpose(empty_IBIS, [0, 2, 1])
    data_cube[*, *, *, ii] = empty_IBIS


ENDFOR

save, data_cube, hdr_aa, filename="IRIS_raster_IBIS_spectra.sav"
END
