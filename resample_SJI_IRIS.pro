iris_data = readfits("/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Apr_2017_data/" $ 
+ "iris_l2_20170423_152532_3624505432_SJI_2796_t000.fits")

frame_offset = 15
shifts = fltarr(2, 8)
start_index_SJI = 200
num_SJI_steps = 300

iris_SJI_fin = fltarr(1000, 1000, num_SJI_steps)

for el=0,(num_SJI_steps-1) DO BEGIN 
    iris_win = resample_SJI(iris_data, start_index_SJI+8*el, 8)
    empty = fltarr(978, 978)
    empty[241:690, 343:741] = iris_win 
    empty1 = empty[220:798, 220:798]
    empty_IBIS = CONGRID(empty1, 1000, 1000)

    iris_SJI_fin[*, *, el] = empty_IBIS    
    print, el,"Out of ", num_SJI_steps
ENDFOR

save, iris_SJI_fin, filename="IRIS_SJI_raster_aligned_fin.sav"
END
