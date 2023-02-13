; Compute the Mn I line Doppler velocity from the IRIS rasters
; where we cut out the Mn I 2801.95 line from the Mg II k&h rasters 


;dir_in = "../IRIS_data/CH_data/"
;file_name_in = "iris_l2_20131117_111245_3803010103_raster_t000_r00000.fits"

data_dir = "/Users/molnarad/CU_Boulder/Work/Chromospheric_business/IRIS_waves/IRIS_data/Plage_data/"
files = find_file(data_dir + "*20160102*.fits", count=n_files)

el = 0

line_index = [2]
mu_angle = ["_0.10"]
;line_index = [7, 2, 7, 7, 7, 7, 2, 2, 7] plage
;mu_angle = ["_0.44", "_0.67", "_0.87", "_0.48", "_1.00", "_1.000", "_0.39", "_0.22", "_0.70"]

; line_index = [7, 6, 7, 7, 7, 6] ; CH data
; mu_angle  = ["_0.39", "_0.41", "_0.25", "_0.74", "_0.82", "_0.92"]

file_name_in = files[el]

d = iris_obj(file_name_in)
d->show_lines

line_name = "MnI_2801.9"

;line = 0
;line_index = 6

line_name_MnI  = "MnI_2801.9"
line_name_SiIV = "SiIV_1393"
line_name_CII  = "CII_1335"

IF line_name eq line_name_MnI THEN rest_wave = 2801.9 ELSE $
IF line_name eq line_name_SiIV THEN rest_wave = 1335.71 ELSE $
IF line_name eq line_name_CII THEN rest_wave = 1393.755

; IF line_name eq line_name_MnI THEN line_index = 6 ELSE $
; IF line_name eq line_name_SiIV THEN line_index = 2 ELSE $
; IF line_name eq line_name_CII THEN line_index = 0


waves = d->getlam(line_index[el])
print, waves
data  = d->getvar(line_index[el], /load)

;start_index = 165
start_index = 335
;start_index = 212 
;end_index   = 179
end_index = 355
;end_index = 228

waves = waves[start_index:end_index]
data = data[start_index:end_index, *, *]

size_data = size(data)

n_waves    = size_data[0]
n_slit_pts = size_data[2] 
n_tsteps   = size_data[3]

N_TERMS = 5
bin_factor = 1

_ = min(abs(waves- rest_wave), rest_wave_index) 

n_measurements = n_slit_pts / bin_factor 

fit_params = fltarr(n_tsteps, n_measurements, N_TERMS)
waves1 = waves - waves[0]

FOR ii=0, (n_tsteps-1) DO BEGIN
    FOR jj=0, (n_measurements-2) DO BEGIN
;FOR ii=100, 105 DO BEGIN
;  FOR jj=102, 105 DO BEGIN

        ;I =data[*, jj*bin_factor:jj*bin_factor $
        ;       + bin_factor - 1, ii]
        I = data[*, jj, ii]
        I_max_estimate = min(I, wave_max_estimate) - I[0] ;min for absorption line
                                                          ;max for emission line
        wave_max_estimate = waves1[wave_max_estimate]
        ;A4_estimate = (I[0]-I[-1])/(waves[0]-waves[-1])
        IF (abs(rest_wave-wave_max_estimate) gt 1.0) THEN BEGIN
            ;print,"Wavelength too different" 
            wave_max_estimate = waves1[rest_wave_index]
            I_max_estimate = min(data[*, jj, ii]) - data[0, jj, ii]
        ENDIF
        slope_est = (I[-1] - I[0]) / (waves1[-1] - waves1[0])

        yfit = gaussfit(waves1, I,$
               A, NTERMS=N_TERMS, $
               ESTIMATES=[I_max_estimate, wave_max_estimate, $
                          .01, I[0], slope_est])
        fit_params[ii, jj, *] = A
    ENDFOR
    print, "Timestep ",ii, " done!"
ENDFOR
; yfit = gaussfit(waves, data[])

; file_name_out = file_name_in.Remove(-5) + line_name + ".sav"
file_name_out = data_dir + line_name + mu_angle[el] + ".sav"
save, fit_params, filename=file_name_out

END
