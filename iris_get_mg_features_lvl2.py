"""
Author: Vishal Upendran
Contact: uvishal1995@gmail.com

This code finds Mg II h and k features, and is based on Leenarts et al (Paper II) 2013.

Reference IDL code: https://hesperia.gsfc.nasa.gov/ssw/iris/idl/uio/utils/iris_get_mg_features_lev2.pro

Changelog:
v1.0: Added all the codes for feature identification.
v1.1: Removed Halo beautification, and added flag for multiprocessing
"""

import numpy as np
import numpy as np
import iris_lmsalpy as iris
from astropy.wcs import WCS
from glob import glob
try:
    import multiprocessing
    POOL_FLAG = True
except:
    POOL_FLAG = False

## Some helper functions
def Nearestind(val,x):
    return np.argmin(np.abs(val-x))
def dopp2wav(vel,ref):
    return vel*ref/299792.458+ref
def wav2dopp(wav,ref):
    return 299792.458*(wav-ref)/ref

def FitParabola_max(args):
    wave = args[0]
    spec = args[1]
    center = args[2]
    #No of points for parabola fit
    N = 3
    minval = np.max([0,center-N])
    maxval = np.min([len(wave)-1,center+N])
    #Parabola fit
    Spec_of_wave = np.poly1d(np.polyfit(wave[minval:maxval],spec[minval:maxval],2))
    #Inference
    new_wave = np.linspace(wave[minval],wave[maxval],50)
    spec_fit = Spec_of_wave(new_wave)
    mloc = new_wave[np.argmax(spec_fit)]
    return [wav2dopp(mloc,BASE),np.max(spec_fit)]

def FitParabola_min(args):
    wave = args[0]
    spec = args[1]
    center = args[2]
    #No of points for parabola fit
    N = 3
    #Parabola fit
    minval = np.max([0,center-N])
    maxval = np.min([len(wave)-1,center+N])
    #Parabola fit
    Spec_of_wave = np.poly1d(np.polyfit(wave[minval:maxval],spec[minval:maxval],2))
    #Inference
    new_wave = np.linspace(wave[minval],wave[maxval],50)
    spec_fit = Spec_of_wave(new_wave)
    mloc = new_wave[np.argmin(spec_fit)]
    return [wav2dopp(mloc,BASE),np.min(spec_fit)]

def Maxmin(d,wave,spec):
    maxv,minv =  np.where((d)<0)[0],np.where((d)>0)[0]
    if len(minv)==0:
        minv=[Nearestind(wave,dopp2wav(5,BASE))]
    else:
        minv=[minv[np.argmin(np.abs(wave[minv]-BASE))]]
    center = FitParabola_min([wave,spec,minv[0]])

    if len(maxv)==0:
        blue = np.asarray([np.nan]*2)
        red = np.asarray([np.nan]*2)
    elif len(maxv)==1:
        if wave[maxv[0]]<dopp2wav(center[0],BASE):
            blue = FitParabola_max([wave,spec,maxv[0]])
            red = np.asarray([np.nan]*2)
        else:
            blue = np.asarray([np.nan]*2)
            red = FitParabola_max([wave,spec,maxv[0]])
    elif len(maxv)==2:
        blue = FitParabola_max([wave,spec,maxv[0]])
        red = FitParabola_max([wave,spec,maxv[-1]])
    else:
        cent = dopp2wav(center[0],BASE)
        b1 = np.where(wave[maxv]<cent)[0]
        r1 = np.where(wave[maxv]>cent)[0]
        if len(r1)!=0:
            red = FitParabola_max([wave,spec,maxv[r1[0]]])
        else:
            red = np.asarray([np.nan]*2)

        if len(b1)!=0:
            blue = FitParabola_max([wave,spec,maxv[b1[-1]]])
        else:
            blue = np.asarray([np.nan]*2)
    return np.asarray([blue,center,red])
def Wrap_maxmin(input_args):
    return Maxmin(input_args[0],input_args[1],input_args[2])

def iris_get_mg_features_lev2(file,vrange=[-40,40],onlyk=False,onlyh=False):
    """ Returns the line center, red peak and blue peaks from Mg II h and k lines.

    Args:
        file (string): path to IRIS raster
        vrange (list, [vmin,vmax]): Velocity range to search for peaks. Defaults to [-40,40].
        onlyk (bool): if True, will only calculate properties for the Mg II k line. Defaults to False.
        onlyh (bool, optional): if True, will only calculate properties for the Mg II h line. Defaults to False.

    Returns:
        lc: 4-D array (line, feature, slit pos., raster pos.) @ line center
        rp: 4-D array (line, feature, slit pos., raster pos.) @ red peak
        bp: 4-D array (line, feature, slit pos., raster pos.) @ blue peak
    """

    Comb_array = []
    rast = iris.extract_irisL2data.load(file,window_info=["Mg II k 2796"],verbose=True)
    mg2_loc = np.where(iris.extract_irisL2data.show_lines(file)=='Mg II k 2796')[0][0]
    extension = mg2_loc+1
    head = iris.extract_irisL2data.only_header(file,extension=extension)
    wcs = WCS(head)
    new_mg = rast.raster['Mg II k 2796'].data
    m_to_nm = 1e9  # convert wavelength to nm
    nwave = new_mg.shape[2]
    wavelength = wcs.all_pix2world(np.arange(nwave), [0.], [0.], 0)[0] * m_to_nm
    if onlyk:
        BASE_LIST = [279.63509493]
    elif onlyh:
        BASE_LIST = [280.35297192]
    else:
        BASE_LIST = [279.63509493,280.35297192]

    for BASE in BASE_LIST:
        #Get the range over which spectral line is present.
        dvp_fit = dopp2wav(vrange[1],BASE)
        dvn_fit = dopp2wav(vrange[0],BASE)
        #Get nearest index from original spectrum to cutout the line
        indp_fit = Nearestind(dvp_fit,wavelength)
        indn_fit = Nearestind(dvn_fit,wavelength)
        # Save temporary variables of wavelength and spectrum
        wave = wavelength[indn_fit-1:indp_fit+1]
        spectrum=new_mg[:,:,indn_fit-1:indp_fit+1]
        print(wave.shape,spectrum.shape,wavelength.shape)
        #Get the derivative
        peakvals = np.gradient(spectrum,axis=-1)
        sign = np.sign(peakvals)
        #Get the maxima and minima
        diff = sign[:,:,1:]-sign[:,:,:-1]
        print(diff.shape,spectrum.shape)

        new_spec = spectrum.reshape([-1,spectrum.shape[-1]])
        new_diff = diff.reshape([-1,diff.shape[-1]])
        input_args=[[new_diff[loc],wave,new_spec[loc]] for loc in np.arange(len(new_spec))]
        if POOL_FLAG:
            pool=multiprocessing.Pool(processes=multiprocessing.cpu_count())
            M=np.asarray(pool.map(Wrap_maxmin,input_args)).reshape([spectrum.shape[0],spectrum.shape[1],3,2])
            pool.close()
        else:
            M=np.asarray([Wrap_maxmin(arg) for arg in input_args]).reshape([spectrum.shape[0],spectrum.shape[1],3,2])
        Comb_array.append(M)

    Comb_array = np.asarray(Comb_array)
    bp = Comb_array[:,0]
    lc = Comb_array[:,1]
    rp = Comb_array[:,2]
    return lc,rp,bp
