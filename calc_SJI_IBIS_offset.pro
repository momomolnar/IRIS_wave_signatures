ca_data = readfits("/Users/molnarad/CU_Boulder/Work/Chromospheric_business/Comps_2/comps_data/IBIS/nb.int.lc.23Apr2017.target2.all.8542.ser_170444.fits", hdr_ca)

iris_data = readfits("/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Apr_2017_data/" $ 
                     + "iris_l2_20170423_152532_3624505432_SJI_2796_t000.fits")


frame_offset = 15
shifts = fltarr(2, 8)
start_index_SJI = 400
num_SJI_steps = 100

for el=0,(num_SJI_steps-1) DO BEGIN 
    iris_win = resample_SJI(iris_data, start_index_SJI+8*el, 8)
    empty = fltarr(978, 978)
    empty[241:690, 343:741] = iris_win 
    empty1 = empty[220:798, 220:798]
    empty_IBIS = CONGRID(empty1, 1000, 1000)


    shifts = xyoff(empty_IBIS, ca_data[*, *, 0], $
                          400, 400)

ENDFOR
END
