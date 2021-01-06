
file_name_in = "../IRIS_data/iris_l2_20131116_073345_3803010103_raster_t000_r00000.fits"

vr = [-40, 40]

d = iris_obj(file_name_in)
d->show_lines

;line = 0
line = 2

IF line eq 2 THEN rest_wave = 1393.755 ELSE $
IF line eq 0 THEN rest_wave = 1335.71   

waves = d->getlam(line)
data  = d->getvar(line, /load)

size_data = size(data)

n_waves    = size_data[1]
n_slit_pts = size_data[2] 
n_tsteps   = size_data[3]

N_TERMS = 4
bin_factor = 3
dwave = 80
_ = min(abs(waves[dwave:-1*dwave] - rest_wave), rest_wave_index) 


n_measurements = n_slit_pts / bin_factor 

fit_params = fltarr(n_tsteps, n_measurements, N_TERMS)
waves1 = waves[dwave:(-1*dwave)]

FOR ii=0, (n_tsteps-1) DO BEGIN
    FOR jj=0, (n_measurements-1) DO BEGIN
        I =median(data[dwave:(-1*dwave), jj*bin_factor:jj*bin_factor $
               + bin_factor - 1, ii], dim=2)
        I_max_estimate = max(I, wave_max_estimate)

        IF (abs(rest_wave-wave_max_estimate) gt .2) THEN BEGIN
            
            wave_max_estimate = rest_wave_index
            I_max_estimate = max(data[dwave+97:dwave+117, jj, ii])
        ENDIF


        yfit = gaussfit(waves1, I,$
               A, NTERMS=N_TERMS, $
               ESTIMATES=[I_max_estimate, waves1[wave_max_estimate], $
                          .1, 2])
        fit_params[ii, jj, *] = A
    ENDFOR
    print, "Timestep ",ii, " done!"
ENDFOR
;yfit = gaussfit(waves, data[])

IF line eq 0 THEN name_add_on = "CII_vel.sav" ELSE $
IF line eq 2 THEN name_add_on = "SiIV_vel.sav"
file_name_out = file_name_in.Remove(-5) + name_add_on

save, fit_params, filename=file_name_out

END
