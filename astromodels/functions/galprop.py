import hashlib
import numpy as np
import astropy.units as u

from astropy.io import fits
from future.utils import with_metaclass
from scipy.interpolate import RegularGridInterpolator as GridInterpolate
from astropy.coordinates import SkyCoord, ICRS, BaseCoordinateFrame
from astromodels.functions.function import Function3D, FunctionMeta
from astromodels.utils.angular_distance import angular_distance_fast

class GalPropTemplate_3D(with_metaclass(FunctionMeta, Function3D)):
    r"""
        description :
            Use a 3D template that has morphology and flux information.
            GalProp, DRAGON or a similar model in fits format would work. 
            Only parameter is a normalization factor. 
        latex : $ K $
        parameters :
            K :
                desc : normalization
                initial value : 1.0
                fix : yes
            hash :
                
                desc : hash of model map [needed for memoization]
                initial value : 1
                fix : yes        

            lon0 :
                
                desc: Longitude of the center of the source
                initial value : 0.0
                min : 0.0
                max : 360.0

            lat0 : 
            
                desc : Latitude of the center of the source
                initial value : 0.0
                min : -90.0
                max : 90.0

    """

    #__metaclass__ = FunctionMeta

    def _set_units(self, x_unit, y_unit, z_unit, w_unit):


        self.lon0.unit = x_unit
        self.lat0.unit = y_unit

        self.K.unit = (u.MeV * u.cm**2 * u.s * u.sr) ** (-1)
        #self.K.unit  = (u.sr)**(-1) #if templates are normalized, use this as the unit

    def _setup(self):

        self._frame = 'ICRS' #ICRS()
        self._map = None
        self._fitsfile = None
        self._interpmap = None

    def set_frame(self, new_frame):
        """
        Set a new frame for the coordinates (the default is ICRS J2000)
        :param new_frame: a coordinate frame from astropy
        :return: (none)
        """
        assert isinstance(new_frame, BaseCoordinateFrame)

        self._frame = new_frame

    def load_file(self, fitsfile, phi1, phi2, theta1, theta2, galactic=False, log_interp=True, ihdu=0):
        
        if fitsfile is None:
        
            raise RuntimeError( "Need to specify a fits file with a template map." )

        self._fitsfile = fitsfile

        p1,p2,t1,t2 = self.define_region(phi1,phi2,theta1,theta2,galactic)

        self.ramin  = p1
        self.ramax  = p2
        self.decmin = t1
        self.decmax = t2
        
        with fits.open(self._fitsfile) as f:

            self._delLon    = f[ihdu].header['CDELT1']
            self._delLat    = f[ihdu].header['CDELT2']
            self._delEn     = 0.2 #0.2 #f[ihdu].header['CDELT3']
            self._refLon    = f[ihdu].header['CRVAL1']
            self._refLat    = f[ihdu].header['CRVAL2']
            self._refEn     = 5 #np.log10(f[ihdu].header['CRVAL3']) # values in log10
            self._map       = f[ihdu].data
            self._refLonPix = f[ihdu].header['CRPIX1'] # reference pixel Longitude
            self._refLatPix = f[ihdu].header['CRPIX2'] # reference pixel Latitude

            self._nl        = f[ihdu].header['NAXIS1'] #longitude
            self._nb        = f[ihdu].header['NAXIS2'] #latitude
            self._ne        = f[ihdu].header['NAXIS3'] #energy


            #Create the function for the interpolation
            self._L = np.linspace(self._refLon - (self._refLonPix) * self._delLon,
                                  self._refLon + (self._nl-self._refLonPix-1) * self._delLon,
                                  self._nl)

            self._B = np.linspace(self._refLat - (self._refLatPix) * self._delLat,
                                  self._refLat + (self._nb-self._refLatPix-1) * self._delLat,
                                  self._nb)

            self._E = np.linspace(self._refEn, self._refEn+(self._ne-1)*self._delEn, self._ne)

            #print(np.log10(self._E))
            #print("self._E looks like: {0}".format(self._E))
            #for i in xrange(len(self._E)):
            #    self._map[i] = self._map[i].to(self.K.unit)/(np.power(10,self._E[i])*np.power(10,self._E[i])) # Map units in Mev / cm^2 s sr, changing to 1 / MeV cm^2 s sr
                #self._map[i] = (np.fliplr(self._map[i]))

            #interpolation is carried using the log scale by default as it produces better results
            # this can be disabled by changing the parameter log_interp to False 
            if log_interp:
            
                self._F = GridInterpolate((self._E,self._B,self._L), np.log10(self._map),
                                            method="linear", bounds_error=False, fill_value=0.0)

                self._is_log10 = True
            
            else:
            
                self._F = GridInterpolate((self._E,self._B,self._L), self._map,
                                            method="linear", bounds_error=False,fill_value=0.0)

                self._is_log10 = False
            
            h = hashlib.sha224()
            h.update( self._map )
            self.hash = int(h.hexdigest(), 16)

    def to_dict(self, minimal=False):
        
        data = super(Function3D, self).to_dict(minimal)

        if not minimal:
        
            data['extra_setup'] = {
                                    "_fitsfile":self._fitsfile, 
                                    "_frame": self._frame, 
                                    "ramin": self.ramin, 
                                    "ramax": self.ramax, 
                                    "decmin": self.decmin, 
                                    "decmax":self.decmax
                                  }

        return data

    def which_model_file(self):
        
        return self._fitsfile


    def evaluate(self, x, y, z, K, hash, lon0, lat0):

        if self._map is None:
    
            self.load_file(self._fitsfile, self.ramin, self.ramax,
                        self.decmin, self.decmax, False, True, ihdu=0)

        #transform energy from keV to MeV 
        # galprop likes MeV, 3ML likes keV
        convert_val = np.log10((u.MeV.to('keV')/u.keV).value)

        # Interpolated values can be cached since we are fitting the constant K
        if self._interpmap is None:

            # We assume x and y are R.A. and Dec
            #_coord = SkyCoord(ra=x, dec=y, frame=self._frame, unit="deg")
            #b = _coord.transform_to('galactic').b.value
            #l = _coord.transform_to('galactic').l.value

            lon = x
            lat = y

            #self.lon0.value = self._refLon
            #self.lat0.value = self._refLat
            angsep = angular_distance_fast(lon0, lat0, lon, lat)

            # if only one energy is passed, make sure we can iterate just once
            if not isinstance(z, np.ndarray):
            
                z=np.array(z)

            #print("raw_energy:")
            #print(z)
            energy = np.log10(z) - convert_val
            #print("converted_energy:")
            #print(energy)


            if lon.size != lat.size:
                
                raise AttributeError("Lon and Lat should be the same size")


            f_new=np.zeros([energy.size, lat.size]) # Nicola: The image is a one dimensional array

            print("Building Interpolation map....")
            
            num = lat.size

            for i,e in enumerate(energy):
                #print("Doing Energy E={0}".format(e))

                engs = np.repeat(e, num)
                
                slice_points = tuple((engs, lat, lon))
                
                if self._is_log10:

                    #if interpolation of flux is carried using log10 scale
                    #ensure values outside function's range remain zero, changing to linear scale
                    #makes this values evaluate to 1.
                    #i.e. if function evaluates to zero -> 10**(0) = 1. This makes the fit fail in 3ML.

                    log_slice = self._F(slice_points)

                    flux_slice = np.array([0. if x==0. else np.power(10.,x)
                                        for x in log_slice])

                else:

                    flux_slice = self._F(slice_points)

                f_new[i] = flux_slice
                    
            print("Finished building interpolation map!")
            
            assert np.all(np.isfinite(f_new)), (
                "some interpolated values are wrong")
            
            self._interpmap=f_new

        A = np.multiply(K,self._interpmap/(10**convert_val)) #(1000 is to change from MeV to KeV)
        #A = np.multiply(K,self._interpmap) #(1000 is to change from MeV to KeV)
        
        return A.T


    def define_region(self, a, b, c, d, galactic=False):
        
        if galactic:
        
            lmin = a
            lmax = b
            bmin = c
            bmax = d

            _coord = SkyCoord(l=[lmin,lmin,lmax,lmax],
                            b=[bmin, bmax, bmax, bmin],
                            frame='galactic',
                            unit='deg')
            
            ramin = min(_coord.transform_to('icrs').ra.value)
            ramax = max(_coord.transform_to('icrs').ra.value)
            decmin = min(_coord.transform_to('icrs').dec.value)
            decmax = max(_coord.transform_to('icrs').dec.value)

        else: 

            ramin = a
            ramax = b
            decmin = c
            decmax = d

        return ramin,ramax,decmin,decmax

    def get_boundaries(self):

        min_longitude = self.ramin
        max_longitude = self.ramax
        min_latitude = self.decmin
        max_latitude = self.decmax

        return (min_longitude, max_longitude), (min_latitude, max_latitude)