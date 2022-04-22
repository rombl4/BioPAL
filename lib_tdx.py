import sys
sys.path.append('./dep_gedi_tdx/demgen')

import gdal
import numpy as np
import logging
import xml.etree.ElementTree as ElementTree
import matplotlib.pyplot as plt
from scipy import constants,interpolate,ndimage,special,stats,signal,optimize
from lib import demgen
from pathlib import Path
from lib import geolib
from lib.newtonbackgeo import NewtonBackgeocoder
import xml.etree.cElementTree as et
from matplotlib import cm
from numba import jit
from lib.demgen import tandemhandler
import os
import matplotlib
import psutil
import lib_profile
import lib_plots
import struct
import lib_filter

import socket
if socket.gethostname() == 'hr-slx012':
    matplotlib.use('Agg')


#Definition of constants for invalid values
INVALID_WATER = -1000
INVALID_SETTLEMENTS = -1001
INVALID_NON_FOREST = -1002
INVALID_KZ = -1003


class GEDI_Exception(Exception):
    pass



def read_cos(path_image,parameters):
    """Read the .cos files from TSX/TDX

       The calibration factor is not applied np.sqrt(calFactor)

    Parameters
    ----------
    path_image : str
        Complete path of the image to read (without including the folder IMAGEDATA in the path)
    parameters : dict
        Dictionary with parameters of the master image
        Output of function get_params_from_xml()

    Returns
    -------
        im : 2D numpy array

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    #logging info
    log = logging.getLogger('read_cos')

    #generate file name
    file_name = path_image+'/IMAGEDATA/'+'IMAGE_'+parameters['polLayer']+'_'+parameters['antennaReceiveConfiguration']+'_'+parameters['elevationBeamConfiguration']+'.cos'

    log.info('Reading '+file_name)

    #get the image with gdal as a complex64 (by default using gdal)
    im_gdal = gdal.Open(file_name)
    im = im_gdal.GetRasterBand(1).ReadAsArray()

    #Transform the real and imaginary parts as int16 because
    # each pixel is 2 bytes (int16) real + 2 bytes (int16) imaginry
    im_real = np.asarray(im.real,np.int16)
    im_imag = np.asarray(im.imag,np.int16)

    #Re-intrept each pxiel as a IEEE 754 half-precision binary floating point
    im_real = np.frombuffer(im_real, np.float16)
    im_imag = np.frombuffer(im_imag, np.float16)

    #Reshape to the dimensions of the image
    im_real = im_real.reshape(im.shape)
    im_imag = im_imag.reshape(im.shape)

    #generate the complex image
    im = im_real + 1j*im_imag

    #to apply the calibration factor (get the value from the xml file 'calfactor') then Multiply im by sqrt(parameters['calFactor'])
    return im

def get_params_from_xml(path_acquisition):
    """Extract information from the xml files of the TDX/TSX images

    Parameters
    ----------
    path_acquisition : srt
        String with the complete path of the TDX image

    Returns
    -------
    parameters : dict
        Dictionary with parameters of the master image
    parameters_slave : dict
        Dictionary with parameters of the slave image

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('get_params_from_xml')
    log.info('Getting parameters from xml ...')

    #define a dictionary output to save the parameters as a dictionary
    parameters = {}

    # --- Get  paramaters from the global xml ----

    #Get xml name from the folder path name
    xml_file = path_acquisition.rsplit('/')[-2]+'.xml'
    #parse the xml
    tree = ElementTree.parse(path_acquisition+xml_file)
    root = tree.getroot()

    #### Get the paths of each image

    #go until the level of each name
    productComponents = root.find("productComponents")
    for component in productComponents.findall("component"):
        if component.attrib['componentClass'] == 'imageData':
            #get the names inside the imagedata component
            for i_name in component.findall(".//name"):
                #check if the name corresponds to the folder name (it stats with T)
                if i_name.text[0] == 'T':
                    parameters[i_name.text[0:3]+'_name'] = i_name.text


    ## get active satelite
    parameters['active_sat'] = root.find('.//satelliteID' + root.find('.//inSARmasterID').text.lower()).text[0:3]

    #### get the polarization
    parameters['polLayer'] = root.find(".//polLayer").text

    ### get the antenna receive configuration
    parameters['antennaReceiveConfiguration'] = root.find(".//antennaReceiveConfiguration").text

    ## get the elevation beam configuration
    parameters['elevationBeamConfiguration'] = root.find(".//elevationBeamConfiguration").text

    ## Tap Take id: acquisitionItemID appliedToID processingPriority operationsTypeSat1 operationsTypeSat2
    parameters['tap_take_id'] = root.find(".//TAPtakeID").text
    
    

    # --- Get  paramateres from the xml of the active image ----

    ### get orbit from the active sensor

    #Get xml name from the folder path name
    xml_file = path_acquisition+parameters[parameters['active_sat']+'_name']+'/'+parameters[parameters['active_sat']+'_name']+'.xml'
    #parse the xml
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()

    time_gps_start = np.double(root.find("productInfo").find("sceneInfo").find("start").find(".//timeGPS").text)
    time_gps_start_fraction = np.double(root.find("productInfo").find("sceneInfo").find("start").find(".//timeGPSFraction").text)
    time_ref = time_gps_start + time_gps_start_fraction

    time_orbit = []
    posX = []
    posY = []
    posZ = []
    VelX = []
    VelY = []
    VelZ = []

    for ElementStateVec in root.find("platform").find("orbit").findall("stateVec"):
        time_orbit.append(np.double(ElementStateVec.find(".//timeGPS").text)+np.double(ElementStateVec.find(".//timeGPSFraction").text))
        posX.append(np.double(ElementStateVec.find(".//posX").text))
        posY.append(np.double(ElementStateVec.find(".//posY").text))
        posZ.append(np.double(ElementStateVec.find(".//posZ").text))
        VelX.append(np.double(ElementStateVec.find(".//velX").text))
        VelY.append(np.double(ElementStateVec.find(".//velY").text))
        VelZ.append(np.double(ElementStateVec.find(".//velZ").text))

    orbit = np.zeros((np.size(time_orbit),7))
    orbit[:,0] = np.asarray(time_orbit) - time_ref
    orbit[:,1] = np.asarray(posX)
    orbit[:,2] = np.asarray(posY)
    orbit[:,3] = np.asarray(posZ)
    orbit[:,4] = np.asarray(VelX)
    orbit[:,5] = np.asarray(VelY)
    orbit[:,6] = np.asarray(VelZ)

    parameters['orbit_no_interp'] = orbit

    # get range sampling frequency
    parameters['orbit_direction'] = root.find('productInfo').find('missionInfo').find(".//orbitDirection").text.lower()

    # get range sampling frequency
    parameters['rg_sampling_freq'] = np.double(root.find(".//commonRSF").text)

    #get range delay of the image
    parameters['rg_delay'] = np.double(root.find('productInfo').find('sceneInfo').find('rangeTime').find(".//firstPixel").text)

    #get the sampling of the focused image 'commonPRF'
    parameters['commonPRF'] = np.double(root.find(".//commonPRF").text)

    #number of rows (azimuth)
    parameters['n_az'] = np.double(root.find('productInfo').find('imageDataInfo').find('imageRaster').find(".//numberOfRows").text)
    #number of cols (range)
    parameters['n_rg'] = np.double(root.find('productInfo').find('imageDataInfo').find('imageRaster').find(".//numberOfColumns").text)

    #row spacing
    parameters['spacing_rg'] = np.double(root.find('productInfo').find('imageDataInfo').find('imageRaster').find(".//rowSpacing").text)*constants.c/2.0

    #row spacing
    parameters['spacing_az'] = np.double(root.find('productInfo').find('imageDataInfo').find('imageRaster').find(".//columnSpacing").text)*constants.c/2.0

    # get the ground Range spacing
    parameters['groundNear'] = np.double(root.find("productSpecific").find(".//groundNear").text)
    parameters['groundFar'] = np.double(root.find("productSpecific").find(".//groundFar").text)

    # get the Azimuth spacing
    parameters['projectedSpacingAzimuth'] = np.double(root.find("productSpecific").find(".//projectedSpacingAzimuth").text)

    #effective velocity at mid range (v0 of TAXI)
    parameters['effective_vel_mid_rg'] = np.sqrt(np.double(root.find('processing').find('geometry').find('velocityParameter').find('velocityParameterPolynomial').find(".//coefficient").text))

    #get range first pixel
    parameters['rg_first_pixel'] = np.double(root.find('productInfo').find('sceneInfo').find('rangeTime').find(".//firstPixel").text) #* constants.c / 2.0

    # range vector
    parameters['range_vec'] = np.linspace(start=parameters['rg_first_pixel'] * constants.c / 2,
                               stop=(parameters['n_rg']-1)*parameters['spacing_rg']+(parameters['rg_first_pixel'] * constants.c / 2),
                               num=int(parameters['n_rg']))

    ## get center of the image
    for i_corner,ElementCorner in enumerate(root.find("productInfo").find("sceneInfo").findall("sceneCenterCoord")):
        parameters['sceneCenter_lat'] = np.double(ElementCorner.find(".//lat").text)
        parameters['sceneCenter_lon'] = np.double(ElementCorner.find(".//lon").text)
        parameters['sceneCenter_incidenceAngle'] = np.double(ElementCorner.find(".//incidenceAngle").text)


    #chose the center of the active satellite
    parameters['center_coord'] = np.asarray([parameters['sceneCenter_lat'],parameters['sceneCenter_lon']])

    #get the look direction
    parameters['look_dir'] = root.find('productInfo').find('acquisitionInfo').find(".//lookDirection").text.lower()

    # center frequency
    parameters['f0'] = np.double(root.find('instrument').find('radarParameters').find(".//centerFrequency").text)

    # Range Look Bandwidth
    parameters['cdw'] = np.double(root.find('processing').find('processingParameter').find(".//rangeLookBandwidth").text)

    # calibration factor
    parameters['calFactor'] = np.double(root.find('calibration').find('calibrationConstant').find(".//calFactor").text)

    # slant range resolution
    parameters['rg_resolution'] = np.double(root.find('productSpecific').find('complexImageInfo').find(".//slantRangeResolution").text)

    # numner of noise records
    parameters['n_noise_records'] = np.double(root.find('noise').find(".//numberOfNoiseRecords").text)

    # Get noise parameters
    parameters['noise_utc'] = []
    parameters['validity_range_min'] = []
    parameters['validity_range_max'] = []
    parameters['reference_point'] = []
    for i_element,ElementImageNoise in enumerate(root.find("noise").findall("imageNoise")):
        parameters['noise_utc'].append(ElementImageNoise.find(".//timeUTC").text)
        parameters['validity_range_min'].append(np.double(ElementImageNoise.find(".//validityRangeMin").text))
        parameters['validity_range_max'].append(np.double(ElementImageNoise.find(".//validityRangeMax").text))
        parameters['reference_point'].append(np.double(ElementImageNoise.find(".//referencePoint").text))
        parameters['noise_polynomial_degree'] = np.double(ElementImageNoise.find(".//polynomialDegree").text)

    #get coeficients
    parameters['noise_coef'] = np.zeros((int(parameters['n_noise_records']),int(parameters['noise_polynomial_degree'])+1))
    for i_element, ElementImageNoise in enumerate(root.find("noise").findall("imageNoise")):
        for j_element,ElementNoiseEstimate in enumerate(ElementImageNoise.find('noiseEstimate').findall(".//coefficient")):
            parameters['noise_coef'][i_element,j_element] = np.double(ElementNoiseEstimate.text)

    #start UTC
    parameters['start_utc'] = root.find('productInfo').find('sceneInfo').find('start').find(".//timeUTC").text
    #stop UTC
    parameters['stop_utc'] = root.find('productInfo').find('sceneInfo').find('stop').find(".//timeUTC").text

    parameters['rg_last_pixel'] = np.double(root.find('productInfo').find('sceneInfo').find('rangeTime').find(".//lastPixel").text)

    #### --- For the four corner coordinates of the master we need to get the info from TDX and TSX -------

    #### TDX #######
    #Get xml name from the folder path name
    xml_file = path_acquisition+parameters['TDX_name']+'/'+parameters['TDX_name']+'.xml'
    #parse the xml
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()
    ## get corners for the image
    for i_corner,ElementCorner in enumerate(root.find("productInfo").find("sceneInfo").findall("sceneCornerCoord")):
        parameters['sceneCorner_lat_'+str(i_corner+1)+'_TDX'] =  np.double(ElementCorner.find(".//lat").text)
        parameters['sceneCorner_lon_'+str(i_corner+1)+'_TDX'] = np.double(ElementCorner.find(".//lon").text)


    #### TSX #######

    #Get xml name from the folder path name
    xml_file = path_acquisition+parameters['TSX_name']+'/'+parameters['TSX_name']+'.xml'
    #parse the xml
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()
    ## get coorners for the image
    for i_corner,ElementCorner in enumerate(root.find("productInfo").find("sceneInfo").findall("sceneCornerCoord")):
        parameters['sceneCorner_lat_'+str(i_corner+1)+'_TSX'] =  np.double(ElementCorner.find(".//lat").text)
        parameters['sceneCorner_lon_'+str(i_corner+1)+'_TSX'] = np.double(ElementCorner.find(".//lon").text)

    ## get corner follwing TAXI procedure
    if parameters['sceneCorner_lat_1_'+parameters['active_sat']] > parameters['sceneCorner_lat_3_'+parameters['active_sat']]:
        parameters['sceneCorner_lat_1'] = np.max([parameters['sceneCorner_lat_1_TDX'],parameters['sceneCorner_lat_1_TSX']])
        parameters['sceneCorner_lat_2'] = np.max([parameters['sceneCorner_lat_2_TDX'],parameters['sceneCorner_lat_2_TSX']])
        parameters['sceneCorner_lat_3'] = np.min([parameters['sceneCorner_lat_3_TDX'],parameters['sceneCorner_lat_3_TSX']])
        parameters['sceneCorner_lat_4'] = np.min([parameters['sceneCorner_lat_4_TDX'],parameters['sceneCorner_lat_4_TSX']])
    else:
        parameters['sceneCorner_lat_1'] = np.min([parameters['sceneCorner_lat_1_TDX'],parameters['sceneCorner_lat_1_TSX']])
        parameters['sceneCorner_lat_2'] = np.min([parameters['sceneCorner_lat_2_TDX'],parameters['sceneCorner_lat_2_TSX']])
        parameters['sceneCorner_lat_3'] = np.max([parameters['sceneCorner_lat_3_TDX'],parameters['sceneCorner_lat_3_TSX']])
        parameters['sceneCorner_lat_4'] = np.max([parameters['sceneCorner_lat_4_TDX'],parameters['sceneCorner_lat_4_TSX']])

    parameters['sceneCorner_lon_1'] = parameters['sceneCorner_lon_1_'+parameters['active_sat']]
    parameters['sceneCorner_lon_2'] = parameters['sceneCorner_lon_2_'+parameters['active_sat']]
    parameters['sceneCorner_lon_3'] = parameters['sceneCorner_lon_3_'+parameters['active_sat']]
    parameters['sceneCorner_lon_4'] = parameters['sceneCorner_lon_4_'+parameters['active_sat']]

    #Position of the corners: upper right, upper left, lower right ,lower left
    corners = np.zeros((4,2),'double')
    corners[0,0] = parameters['sceneCorner_lat_1']
    corners[1,0] = parameters['sceneCorner_lat_2']
    corners[2,0] = parameters['sceneCorner_lat_3']
    corners[3,0] = parameters['sceneCorner_lat_4']
    corners[0,1] = parameters['sceneCorner_lon_1']
    corners[1,1] = parameters['sceneCorner_lon_2']
    corners[2,1] = parameters['sceneCorner_lon_3']
    corners[3,1] = parameters['sceneCorner_lon_4']

    parameters['corners'] = np.copy(corners)
    
    ###### get also parameters of the slave image ##########################################
    
    #define a dictionary output to save the parameters as a dictionary
    parameters_slave = {}
    
    #Get xml name from the folder path name
    slave_name = 'TDX'
    if parameters['active_sat'] == 'TDX': slave_name = 'TSX'
        
    xml_file = path_acquisition+parameters[slave_name+'_name']+'/'+parameters[slave_name+'_name']+'.xml'
    #parse the xml
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()
    
    ## get corners for the image
    for i_corner,ElementCorner in enumerate(root.find("productInfo").find("sceneInfo").findall("sceneCornerCoord")):
        parameters_slave['sceneCorner_lat_'+str(i_corner+1)] =  np.double(ElementCorner.find(".//lat").text)
        parameters_slave['sceneCorner_lon_'+str(i_corner+1)] = np.double(ElementCorner.find(".//lon").text)


    #use same order as with the master
    corners = np.zeros((4,2),'double')
    corners[0,0] = parameters_slave['sceneCorner_lat_1']
    corners[1,0] = parameters_slave['sceneCorner_lat_2']
    corners[2,0] = parameters_slave['sceneCorner_lat_3']
    corners[3,0] = parameters_slave['sceneCorner_lat_4']
    corners[0,1] = parameters_slave['sceneCorner_lon_1']
    corners[1,1] = parameters_slave['sceneCorner_lon_2']
    corners[2,1] = parameters_slave['sceneCorner_lon_3']
    corners[3,1] = parameters_slave['sceneCorner_lon_4']

    parameters_slave['corners'] = np.copy(corners)


    # get orbit slave
    
    time_gps_start = np.double(root.find("productInfo").find("sceneInfo").find("start").find(".//timeGPS").text)
    time_gps_start_fraction = np.double(root.find("productInfo").find("sceneInfo").find("start").find(".//timeGPSFraction").text)
    time_ref = time_gps_start + time_gps_start_fraction
    
    time_orbit = []
    posX = []
    posY = []
    posZ = []
    VelX = []
    VelY = []
    VelZ = []

    for ElementStateVec in root.find("platform").find("orbit").findall("stateVec"):
        time_orbit.append(np.double(ElementStateVec.find(".//timeGPS").text)+np.double(ElementStateVec.find(".//timeGPSFraction").text))
        posX.append(np.double(ElementStateVec.find(".//posX").text))
        posY.append(np.double(ElementStateVec.find(".//posY").text))
        posZ.append(np.double(ElementStateVec.find(".//posZ").text))
        VelX.append(np.double(ElementStateVec.find(".//velX").text))
        VelY.append(np.double(ElementStateVec.find(".//velY").text))
        VelZ.append(np.double(ElementStateVec.find(".//velZ").text))

    orbit = np.zeros((np.size(time_orbit),7))
    orbit[:,0] = np.asarray(time_orbit,'double') - time_ref
    orbit[:,1] = np.asarray(posX)
    orbit[:,2] = np.asarray(posY)
    orbit[:,3] = np.asarray(posZ)
    orbit[:,4] = np.asarray(VelX)
    orbit[:,5] = np.asarray(VelY)
    orbit[:,6] = np.asarray(VelZ)

    parameters_slave['orbit_no_interp'] = orbit


    #get range delay of the image
    parameters_slave['rg_delay'] = np.double(root.find('productInfo').find('sceneInfo').find('rangeTime').find(".//firstPixel").text)

    # Row spacing
    parameters_slave['spacing_rg'] = np.double(root.find('productInfo').find('imageDataInfo').find('imageRaster').find(".//rowSpacing").text)*constants.c/2.0

    # get the Azimuth spacing
    parameters_slave['projectedSpacingAzimuth'] = np.double(root.find("productSpecific").find(".//projectedSpacingAzimuth").text)

    parameters_slave['groundNear'] = np.double(root.find("productSpecific").find(".//groundNear").text)
    parameters_slave['groundFar'] = np.double(root.find("productSpecific").find(".//groundFar").text)

    #get the sampling of the focused image 'commonPRF'
    parameters_slave['commonPRF'] = np.double(root.find(".//commonPRF").text)

    # calibration factor
    parameters_slave['calFactor'] = np.double(root.find('calibration').find('calibrationConstant').find(".//calFactor").text)

    #number of rows (azimuth)
    parameters_slave['n_az'] = np.double(root.find('productInfo').find('imageDataInfo').find('imageRaster').find(".//numberOfRows").text)
    #number of cols (range)
    parameters_slave['n_rg'] = np.double(root.find('productInfo').find('imageDataInfo').find('imageRaster').find(".//numberOfColumns").text)

    #effective velocity at mid range (v0 of TAXI)
    parameters_slave['effective_vel_mid_rg'] = np.sqrt(np.double(root.find('processing').find('geometry').find('velocityParameter').find('velocityParameterPolynomial').find(".//coefficient").text))

    # get range sampling frequency
    parameters_slave['rg_sampling_freq'] = np.double(root.find(".//commonRSF").text)

    #get the look direction
    parameters_slave['look_dir'] = root.find('productInfo').find('acquisitionInfo').find(".//lookDirection").text.lower()

    #center frequency
    parameters_slave['f0'] = np.double(root.find('instrument').find('radarParameters').find(".//centerFrequency").text)

    # numner of noise records
    parameters_slave['n_noise_records'] = np.double(root.find('noise').find(".//numberOfNoiseRecords").text)

    # Get noise parameters
    parameters_slave['noise_utc'] = []
    parameters_slave['validity_range_min'] = []
    parameters_slave['validity_range_max'] = []
    parameters_slave['reference_point'] = []
    for i_element,ElementImageNoise in enumerate(root.find("noise").findall("imageNoise")):
        parameters_slave['noise_utc'].append(ElementImageNoise.find(".//timeUTC").text)
        parameters_slave['validity_range_min'].append(np.double(ElementImageNoise.find(".//validityRangeMin").text))
        parameters_slave['validity_range_max'].append(np.double(ElementImageNoise.find(".//validityRangeMax").text))
        parameters_slave['reference_point'].append(np.double(ElementImageNoise.find(".//referencePoint").text))
        parameters_slave['noise_polynomial_degree'] = np.double(ElementImageNoise.find(".//polynomialDegree").text)

    #get coeficients
    parameters_slave['noise_coef'] = np.zeros((int(parameters_slave['n_noise_records']),int(parameters_slave['noise_polynomial_degree'])+1))
    for i_element, ElementImageNoise in enumerate(root.find("noise").findall("imageNoise")):
        for j_element,ElementNoiseEstimate in enumerate(ElementImageNoise.find('noiseEstimate').findall(".//coefficient")):
            parameters_slave['noise_coef'][i_element,j_element] = np.double(ElementNoiseEstimate.text)

    #start UTC
    parameters_slave['start_utc'] = root.find('productInfo').find('sceneInfo').find('start').find(".//timeUTC").text
    #stop UTC
    parameters_slave['stop_utc'] = root.find('productInfo').find('sceneInfo').find('stop').find(".//timeUTC").text

    parameters_slave['rg_last_pixel'] = np.double(root.find('productInfo').find('sceneInfo').find('rangeTime').find(".//lastPixel").text)

    #get range first pixel
    parameters_slave['rg_first_pixel'] = np.double(root.find('productInfo').find('sceneInfo').find('rangeTime').find(".//firstPixel").text) #* constants.c / 2.0

    # range vector
    parameters_slave['range_vec'] = np.linspace(start=parameters_slave['rg_first_pixel'] * constants.c / 2,
                               stop=(parameters_slave['n_rg']-1)*parameters_slave['spacing_rg']+(parameters_slave['rg_first_pixel'] * constants.c / 2),
                               num=int(parameters_slave['n_rg']))


    return parameters,parameters_slave

def compute_interferogram(resolution,parameters,mst,slv,flat,kz,dem,multilook=None,resolution_slcs=1):
    """Computation of the interferogram and coherence

    Parameters
    ----------
    resolution : float
        Desired resolution, to be used in the averaging of the SLC's
    parameters
    mst : 2D numpy array
        SLC master image
        output from read_cos()
    slv : 2D numpy array
        SLC slave image image
        output from read_cos()
    flat : 2D numpy array
        Output from compute_slant_phase_flat() with the SLC dimensions
    kz : 2D numpy array
        Vertical wavenumber
        Ouput from get_baselines() with the SLC dimensions
    dem : 2D numpy array
        dem in range/azimuth coordinates with the same size as SLC
    multilook : list with the pixel size to use for the multilook in azimuth and range
        if it is None we take the size based on the input resolution
    resolution_slcs : int
        resoution at which we smooth the input images before the coherence computation,
        if it is = 1 we do nothing

    Returns
    -------
    interferogram : 2D numpy array
    coherence : 2D numpy array

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('compute_interferogram')
    log.info('Computation of the interferogram and coherence ...')


    # check if the dem has nans
    # NOTE: For the full processing, the dem has no nanas as it is checked in the function get_dem()
    if np.sum(np.isnan(dem)) > 0:
        log.info('Interpolation nan values of the dem')
        points_nan = np.where(np.isnan(dem))
        size_block = 2
        for i_nan_row, i_nan_col in zip(points_nan[0], points_nan[1]):
            value_inter = np.nan
            while np.isnan(value_inter):
                # get the points of the block to interpolate and check that it is inside the image
                size_block = size_block + 2
                ini_row = np.clip(i_nan_row - size_block, 0, dem.shape[0])
                fin_row = np.clip(i_nan_row + size_block, 0, dem.shape[0])
                if ini_row == fin_row: fin_row + 2
                ini_col = np.clip(i_nan_col - size_block, 0, dem.shape[1])
                fin_col = np.clip(i_nan_col + size_block, 0, dem.shape[1])
                if ini_col == fin_col: fin_col + 2
                # we use the mean over the neihboors as interpolator
                value_inter = np.nanmean(dem[ini_row:fin_row, ini_col:fin_col])

            # change the nan value by the interpolated one
            dem[i_nan_row, i_nan_col] = value_inter


    if resolution_slcs > 1:
        log.info('Reduce resolution of SLCs to avoid bias in the coherence')
        ml_rg_slc = np.int(np.round(resolution_slcs / ((parameters['groundNear'] + parameters['groundFar']) / 2.)))
        ml_az_slc = np.int(np.round(resolution_slcs / parameters['projectedSpacingAzimuth']))
        # smooth images
        mst = ndimage.uniform_filter(mst.real, (ml_az_slc, ml_rg_slc)) + 1j * ndimage.uniform_filter(mst.imag, (ml_az_slc, ml_rg_slc))
        slv = ndimage.uniform_filter(slv.real, (ml_az_slc, ml_rg_slc)) + 1j * ndimage.uniform_filter(slv.imag, (ml_az_slc, ml_rg_slc))


    #get multi-look size in radar coordinates. For the range we get the mean value between near and far
    if multilook is None:
        ml_rg = np.int(np.round(resolution/((parameters['groundNear']+parameters['groundFar'])/2.)))
        ml_az = np.int(np.round(resolution/parameters['projectedSpacingAzimuth']))
    else:
        ml_az = multilook[0]
        ml_rg = multilook[1]

    log.info('Multilook in range ='+str(ml_rg))
    log.info('Multilook in azimuth ='+str(ml_az))


    ####make fast version using multyply and output keyword
    #interferofram * flateart_phase * topographic phase
    log.info(' Generate interferogram step 1 of 2 ...')
    interferogram = np.multiply(mst,np.conj(slv))
    #interferogram = numexpr.evaluate("mst*conj(slv)")
    # NOTE: Check if we need (or not) to remove the meanheight from the dem before the topograhic phase compensation
    # np.multiply(np.multiply(interferogram, np.exp(-flat * 1j), out=interferogram), np.exp(np.multiply(-(dem-meanheight), kz * 1j)), out=interferogram)
    np.multiply(np.multiply(interferogram,np.exp(-flat*1j),out=interferogram),np.exp(np.multiply(-dem,kz*1j)),out=interferogram)
    #interferogram = numexpr.evaluate("interferogram*exp(-flat*1j)*exp(-dem*kz*1j)")


    log.info(' Generate interferogram step 2 of 2 ...')
    ndimage.uniform_filter(interferogram.real,(ml_az,ml_rg),output=interferogram.real)+1j*ndimage.uniform_filter(interferogram.imag,(ml_az, ml_rg),output=interferogram.imag)

    # #coherence
    log.info(' Generate coherence step 1 of 3 ...')
    mst_multilook = np.multiply(mst,np.conj(mst))
    #mst_multilook = numexpr.evaluate("mst*conj(mst)")
    ndimage.uniform_filter(mst_multilook.real,(ml_az,ml_rg),output=mst_multilook.real)+1j*ndimage.uniform_filter(mst_multilook.imag,(ml_az, ml_rg),output=mst_multilook.imag)

    log.info(' Generate coherence step 2 of 3 ...')
    slv_multilook = np.multiply(slv, np.conj(slv))
    #slv_multilook = numexpr.evaluate("slv*conj(slv)")
    ndimage.uniform_filter(slv_multilook.real, (ml_az, ml_rg), output=slv_multilook.real) + 1j * ndimage.uniform_filter(slv_multilook.imag, (ml_az, ml_rg), output=slv_multilook.imag)

    log.info(' Generate coherence step 3 of 3 ...')
    coherence = np.divide(interferogram,np.sqrt(np.multiply(mst_multilook,slv_multilook)))
    #coherence = numexpr.evaluate("(interferogram)/(sqrt(mst_multilook*slv_multilook))")

    log.info('Computation of the interferogram and coherence ok!')

    return interferogram,coherence


def get_2dlut_kz_coh_height_from_master_profile(master_profile,height_vector,kz_min=0,kz_max=0.5,n_elements_lut=100000):
    """Generate a LUT table that realtes the values of kz and height with coherence

    Parameters
    ----------
    master_profile : 1D numpy array
        Common profile generated from GEDI data
    height_vector: module
        height vector used for the LUT
    kz_min : float,optional
        Minimum value of kz for the LUT
    kz_max : float, optional
        Maximum value of kz for the LUT
    n_elements_lut : int,optional
        size of the lut for the precessiong to comp

    Returns
    -------
    lut_2d_kz_coh : 2D numpy array
        With dimensions (n_elements_lut,size_common_profile)
    kz_lut_axes : 1D numpy array
        Defines the rows of lut_2d_kz_coh
        kz_lut_axes = np.linspace(kz_min, kz_max, n_elements_lut)

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """


    log = logging.getLogger('get_2dlut_kz_coh_height_from_master_profile')
    log.info('Computing 2D LUT for kz and height <-> coherence')

    kz_lut_axes = np.linspace(kz_min, kz_max, n_elements_lut)

    # #generation a 2D LUT for a range of kz values
    if len(height_vector) != len(master_profile):
        master_profile = ndimage.zoom(master_profile,np.float(len(height_vector))/len(master_profile))

    size_common_profile = len(master_profile)
    arrray_from_0_to_1 = np.linspace(0, 1, num=size_common_profile)
    aux_mat = np.matmul(np.reshape(arrray_from_0_to_1, (size_common_profile, 1)),np.reshape(height_vector, (1, len(height_vector))))
    constant_sum_master_profile = np.abs(np.sum(master_profile))


    lut_2d_kz_coh = np.zeros((len(kz_lut_axes),size_common_profile))
    for i_kz,kz_value in enumerate(kz_lut_axes):
        exp_kz_mat = np.exp(0 + 1j * kz_value * aux_mat)
        aux_mat2 = np.squeeze(np.abs(np.matmul(exp_kz_mat, np.reshape(master_profile, (size_common_profile, 1)))))
        lut_2d_kz_coh[i_kz,:] = aux_mat2 / constant_sum_master_profile


    return lut_2d_kz_coh,kz_lut_axes

@jit(nopython=True)
def one_pixel_forest_height(kz_cor_pixel,kz_lut_axes,lut_kz_coh_heights,coh_cor_pixel,height_vector):
    """Compute the forest height for one pixel

    Parameters
    ----------
    kz_cor_pixel : float
        Corrected kz for one pixel
    kz_lut_axes : 1D numpy array
        All possible values in the LUT
    lut_kz_coh_heights : 2D numpy array
        LUT taht realtes kz and coherence with ehgiht
    coh_cor_pixel : float
        Corrected coherence for one pixel
    height_vector : 1D numpy array
        Vector of heights

    Returns
    -------
    height_vector[pos_lut] : float, value of height


    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """
    #  Use a pre-generated LUT for certain range of kzs
    pos_kz = np.argmin(np.abs(kz_cor_pixel - kz_lut_axes))
    lut_coh_heights = lut_kz_coh_heights[pos_kz, :]

    # find the position of the lut which is more similar to the coherence input
    pos_lut = np.argmin(np.abs(coh_cor_pixel - lut_coh_heights))

    return height_vector[pos_lut]


def forest_height_inversion(inputs,kz_cor,coh_cor,parameters,lut_kz_coh_heights=None,kz_lut_axes=None,master_profile=None,use_input_size_kz_coh=False):
    """Make the forest height inversion based on a LUT.

        There are two options to compute the forest heigth:

              1- Use a predefined lut that relates kz, coherence and hieght
                     - Input parameter 'lut_kz_coh_heights' that comes from the function get_2dlut_kz_coh_height_from_master_profile
                     - Input paraneter 'kz_lut_axes': Axes of the lut, it is a input parameter of the function get_2dlut_kz_coh_height_from_master_profile
              2- Compute the lut that relates coherence height for each kz (It is much more slower than option 1)
                     - Input paramenter 'master_profile'. It is the master profile used to compute the lut

              NOTE: The option 1 has high priority, as it is faster than option 2:
                      - It means that, if the user provides as inputs 'lut_kz_coh_heights' and 'master_profile',
                         the function will use the lut_kz_coh_heights to get the heights and ignore the master profile.
                      - If the lut generate with the 'get_2dlut_kz_coh_height_from_master_profile' has enoguh samples (i. e.
                          the 'kz_lut_axes') the results of option 1 and option 2 are (almost) the same.

    Parameters
    ----------
    inputs: module
        Module from the inputs file used in the GEDI/TDX procesinng
        Before calling the function make import inputs
    kz_cor : 2D numpy array
        Kz corrected by the den
        Output from processing_tdx_until_coherence()
    coh_cor : 2D numpy array
        Coherence corrected by the dem
        Output from processing_tdx_until_coherence()
    parameters : dict
        Inforamtion realted to the master image
        Output from processing_tdx_until_coherence()
    lut_2d_kz_coh : 2D numpy array
        With dimensions (n_elements_lut,size_common_profile)
        Output from get_2dlut_kz_coh_height_from_master_profile()
    kz_lut_axes : 1D numpy array
        Defines the rows of lut_2d_kz_coh
        kz_lut_axes = np.linspace(kz_min, kz_max, n_elements_lut)
        Output from get_2dlut_kz_coh_height_from_master_profile()
    master_profile : 1D numpy array
        Common profile generated from GEDI data
    use_input_size_kz_coh : Bool
        Flag to not change the size of the input arrays coh and kz

    Returns
    -------
    forest_height : 2D numpy array
        Forest heights

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """


    log = logging.getLogger('forest_height_inversion')
    log.info('Computation of the forest height inversion')

    height_vector = np.linspace(inputs.min_height_vector, inputs.max_height_vector, num=inputs.n_elements_height_vector)

    # INCREASE COHERENCE TO REMOVE BIAS IN THE LOWER PART
    #coh_cor = np.clip(coh_cor / inputs.decorrelation_coherence_before_inversion, 0, 1)

    if use_input_size_kz_coh==False:
        # reduce the coh_cor and kz_cor
        pixel_spacing = inputs.pixel_spacing_out

        log.info('Original pixels range = ' + str(coh_cor.shape[1]))
        log.info('Original pixels azimuth = ' + str(coh_cor.shape[0]))

        factor_rescale_rg = 1.0 / (pixel_spacing / ((parameters['groundNear'] + parameters['groundFar']) / 2.))
        factor_rescale_az = 1.0 / (pixel_spacing / parameters['projectedSpacingAzimuth'])
        coh_cor  = ndimage.zoom(coh_cor, (factor_rescale_az, factor_rescale_rg),order=1)
        kz_cor = ndimage.zoom(kz_cor, (factor_rescale_az, factor_rescale_rg), order=1)

        # new_shape_az = int(kz_cor.shape[0] * factor_rescale_az)+1
        # new_shape_rg = int(kz_cor.shape[1] * factor_rescale_rg)+1
        # new_shape = [new_shape_az,new_shape_rg]
        # kz_cor = rebin_arbitrary_dims(kz_cor, new_shape, method='mean')
        # coh_cor = rebin_arbitrary_dims(coh_cor, new_shape, method='mean')

        log.info('Final pixels to invert in range = ' + str(coh_cor.shape[1]))
        log.info('Final pixels to invert in azimuth = ' + str(coh_cor.shape[0]))

    # #quantization error correction (coherence error of quantization is normally around 3 %)
    # coh_cor = coh_cor/inputs.quantization_error
    # coh_cor[(coh_cor < 0) | (coh_cor > 1)] = np.nan


    n_rows, n_cols = coh_cor.shape


    #Generate output matrix
    forest_height = np.zeros(coh_cor.shape)

    #check if we have a predefined lut
    if lut_kz_coh_heights is not None:
        log.info('Using pre-computed lut for a range of kzs')
        #in that case we fuse option 1, a predefined lut
        flag_compute_lut = False
    elif master_profile is not None:
        log.info('The luts will be computed for each kz')
        #in that case we compute the lut for each kz
        flag_compute_lut = True
    else:
        log.error('Please provide a Master profile to generate the lut for each kz or a 2D LUT that relates kz with coherence and height')

    ##check if we have to compute the lut for each kz
    if flag_compute_lut:
        #matrix and constat that we need to compute inside the loops the lut for each kz
        size_common_profile = len(master_profile)
        arrray_from_0_to_1 = np.linspace(0, 1, num=size_common_profile)
        aux_mat = np.matmul(np.reshape(arrray_from_0_to_1, (size_common_profile, 1)),np.reshape(height_vector, (1, len(height_vector))))
        constant_sum_master_profile = np.abs(np.sum(master_profile))

    ##check if we have to compute the lut for each kz or we have a predefined lut
    if flag_compute_lut:
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                exp_kz_mat = np.exp(0 + 1j * kz_cor[i_row, i_col] * aux_mat)
                aux_mat2 = np.squeeze(np.abs(np.matmul(exp_kz_mat, np.reshape(master_profile, (size_common_profile, 1)))))
                lut_coh_heights = aux_mat2 / constant_sum_master_profile

                # find the position of the lut which is more similar to the coherence input
                pos_lut = np.argmin(np.abs(coh_cor[i_row, i_col] - lut_coh_heights))
                forest_height[i_row, i_col] = height_vector[pos_lut]

    else:
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                #Use a function to called with numba (faster than using the code without the function)
                forest_height[i_row, i_col] = one_pixel_forest_height(kz_cor[i_row, i_col], kz_lut_axes, lut_kz_coh_heights,coh_cor[i_row, i_col],height_vector)

                # if mask_kz_points[i_row,i_col] == 0:
                #     #  Use a pre-generated LUT for certain range of kzs
                #     pos_kz = np.argmin(np.abs(kz_cor[i_row, i_col] - kz_lut_axes))
                #     lut_coh_heights = lut_kz_coh_heights[pos_kz,:]
                #
                #     #find the position of the lut which is more similar to the coherence input
                #     pos_lut = np.argmin(np.abs(coh_cor[i_row, i_col] - lut_coh_heights))
                #     forest_height[i_row, i_col] = height_vector[pos_lut]


    return forest_height


def interpol_orbits(parameters,margin=-2,polDegree=7,pointsBefore=14,pointsAfter=16,reqTime=None,same_coefficients=True,parameters_slave=None):
    """Interpolation of the orbits

    Based on interpolorb.pro and InterpolOrbCalc.pro) from TAXI
    It includes only the Chebyshev approximation calculating the values using the same coefficients.

    Parameters
    ----------
    parameters : dict
        Dictionary with parameters of the master image.
        Output of function get_params_from_xml()
    margin : float,optional
        Time margin in seconds for the interpolation
    polDegree : int, optional
        Degree of polynom to use
    pointsBefore : int, optiontal
        To compute the minimum required array index to keep amount of points (default 14)
    pointsAfter : int, optional
        To compute the maximun required index to keep amount of points (default 16)
    reqTime : 1D array, optional
        It is the array fiven in parameters_slave['orbit'][:,0]
        If ReqTime is an input is becasue we are computing the active_orbit
    same_coefficients : bool, optional
        If True all values shall be calculated with the same coefficients, so use the mid of all requested times (by default true)
        Set to False to compute the active orbit
    parameters_slave : dict, optional
        Dictionary with parameters of the slave image.
        Output of function get_params_from_xml()
        Use this argument only if you also use the reqTime

    Returns
    -------
    orbit : 2D numpy array of dimensions (n,7)
        Contain the orbit information in the form of:
        orbit[:,0]: required time vector used for the interpolation
        orbit[:,1]:  interpolated x position vector
        orbit[:,2]:  interpolated y position vector
        orbit[:,3]:  interpolated z position vector
        orbit[:,4]:  interpolated x velocity vector
        orbit[:,5]:  interpolated y velocity vector
        orbit[:,6]:  interpolated z velocity vector

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('interpol_orbits')
    log.info('Interpolation of orbits ...')


    # - Interpolation of orbits (translation of interpolorb.pro and InterpolOrbCalc.pro) from TAXI
    #- Using Chebyshev approximation

    time = parameters['orbit_no_interp'][:,0]
    posX = parameters['orbit_no_interp'][:,1]
    posY = parameters['orbit_no_interp'][:,2]
    posZ = parameters['orbit_no_interp'][:,3]
    velX = parameters['orbit_no_interp'][:,4]
    velY = parameters['orbit_no_interp'][:,5]
    velZ = parameters['orbit_no_interp'][:,6]

    # Compute required time in case we do not have it as an input.
    # If ReqTime is an input is becasue we are computing the active_orbit and we use:
    #   the reqTime (gave it as input) and time from the slave (we take it fromparameters_slave)
    if reqTime is None:

        ### interpolation of orbits
        ext = np.round(np.abs(margin) * 2 / (1.0 / parameters['commonPRF']))
        # get the required time
        reqTime = np.linspace(start=margin,
                              stop=((ext + parameters['n_az']-1) * (1.0 / parameters['commonPRF']))+ margin,
                              num=np.int(ext + parameters['n_az']))

    else:
        time = parameters_slave['orbit_no_interp'][:,0]


    Size_reqTime = len(reqTime)

    if Size_reqTime <= 0: log.error('The time vector is incorrect.')

    ## get the lowest index
    #cut time, position and velocity array
    minIdx=np.argmin(np.abs(time-np.min(reqTime)))

    if time[minIdx] > np.min(reqTime): minIdx = minIdx-1

    #; minimum required array index keep amount of points
    minIdxNPt=len(time)-1-(pointsAfter+pointsBefore)
    #; find min Idx to use
    minIdx=np.max([np.min([minIdx-pointsBefore,minIdxNPt]),0])

    #get the highest index
    #cut time, position and velocity array
    maxIdx=np.argmin(np.abs(time-np.max(reqTime)))

    if time[maxIdx] > np.max(reqTime): maxIdx = maxIdx - 1

    #; maximun required index to keep amount of points
    maxIdxNPt=minIdx+(pointsAfter+pointsBefore)
    #; find min Idx to use
    maxIdx=np.min([np.max([maxIdx+pointsAfter,maxIdxNPt]),len(time)])

    ## from here part of InterpolOrbCalc.pro

    # cut the inputs to the used time
    time = time[minIdx:maxIdx]
    posX = posX[minIdx:maxIdx]
    posY = posY[minIdx:maxIdx]
    posZ = posZ[minIdx:maxIdx]
    velX = velX[minIdx:maxIdx]
    velY = velY[minIdx:maxIdx]
    velZ = velZ[minIdx:maxIdx]

    #get size of the 'cut' array
    size_time = len(time)

    # make output variable with the interpolated orbit
    orbit = np.zeros((Size_reqTime, 7))

    i_time = 0
    while i_time < Size_reqTime:
        #; all values shall be calculated with the same coefficients, so use the mid of all requested times
        if same_coefficients:
            coeffCenterTime = (np.max(reqTime) - np.min(reqTime)) / 2.0 + np.min(reqTime)
        else:
            coeffCenterTime = reqTime[i_time]
        
        
        aux = np.abs(time-coeffCenterTime)
        idx = np.argmin(aux)
    
        #; for TSX approximation a duration of 5 Minutes is recommended (30 samples * 10 second interval)
        #; wanted minimum array index (default 14)
        minIdxReq=idx-14
        #; wanted maximum array index (default 15)
        maxIdxReq=idx+15
        #; minimum required array index to maintain requested pol degree
        minIdxDegree=size_time-1 - polDegree
        #; find min Idx to use
        minIdx=np.max([np.min([minIdxReq,minIdxDegree]),0])
        #; maximun required index to maintain requested pol degree
        maxIdxDegree=minIdx + polDegree
        #; find max Idx to use
        maxIdx=np.min([np.max([maxIdxReq,maxIdxDegree]), size_time-1])
    
        useTime=time[minIdx:maxIdx+1]
        #; transform to new interval
        reqTimeInt=2.0*(reqTime-np.min(useTime)) / (np.max(useTime)-np.min(useTime)) -1
        #; transform to new interval
        timeInt=2.0*(useTime-np.min(useTime)) / (np.max(useTime)-np.min(useTime)) -1
    
        # check if enough points are available for more or less correct approximation
        if maxIdx-minIdx <= 3:
            log.error('Not enough points to approximation correct! Point available:'+str(maxIdx-minIdx+1)+ '. Required at least 5, but more would improve results.')
      
        if same_coefficients:
            # ; all values will be calculated using the same coefficients
            j_time = Size_reqTime
        else:
            #Find the reqtime indices which can be calculated with the same input array indices and
            #  rememember them for calculation with these coeficients
            finished = 0
            j_time = 1
            while finished == 0 and (i_time+j_time) < Size_reqTime:
                checkindx = np.argmin(np.abs(time - reqTime[i_time+j_time]))
                if checkindx == idx:
                    j_time = j_time + 1
                else:
                    finished = 1

        use_reqTimeInt = np.copy(reqTimeInt[i_time:i_time+j_time])
        
        #; calculate chebyshev coefficients
        degree = np.min([maxIdx-minIdx,polDegree+1])
    
        #Interpolate each of the components
    
        #for posX
        coef = np.polynomial.chebyshev.chebfit(timeInt, posX[minIdx:maxIdx+1],degree)
        orbit[i_time:i_time+j_time,1] =  np.polynomial.chebyshev.chebval(use_reqTimeInt,coef)
    
        #for posY
        coef = np.polynomial.chebyshev.chebfit(timeInt, posY[minIdx:maxIdx+1], degree)
        orbit[i_time:i_time+j_time,2] = np.polynomial.chebyshev.chebval(use_reqTimeInt, coef)
    
        #for posZ
        coef = np.polynomial.chebyshev.chebfit(timeInt, posZ[minIdx:maxIdx+1], degree)
        orbit[i_time:i_time+j_time,3] = np.polynomial.chebyshev.chebval(use_reqTimeInt, coef)
    
        #for velX
        coef = np.polynomial.chebyshev.chebfit(timeInt, velX[minIdx:maxIdx+1],degree)
        orbit[i_time:i_time+j_time,4] =  np.polynomial.chebyshev.chebval(use_reqTimeInt,coef)
    
        #for velY
        coef = np.polynomial.chebyshev.chebfit(timeInt, velY[minIdx:maxIdx+1], degree)
        orbit[i_time:i_time+j_time,5] = np.polynomial.chebyshev.chebval(use_reqTimeInt, coef)
    
        #for velZ
        coef = np.polynomial.chebyshev.chebfit(timeInt, velZ[minIdx:maxIdx+1], degree)
        orbit[i_time:i_time+j_time,6] = np.polynomial.chebyshev.chebval(use_reqTimeInt, coef)

        i_time = i_time + j_time

    #add time
    orbit[:,0] = np.asarray(reqTime)

    return orbit


def get_dem(path_dem,type_dem,parameters,margin_degrees=0.5,NumThreads=5):
    """Get the dem for the processing

    It accepts TanDEM DEM 90 m or SRTM DEM.

    If the SRTM is used, and the data is not available it downlads the data from:
        - dds.cr.usgs.gov  (/srtm/version2_1/SRTM3)

    TanDEM DEM 90 m available on:
        - https://download.geoservice.dlr.de/TDM90/

    Parameters
    ----------
    path_dem : str
        Complete path where the dem
    type_dem : {'TANDEM','SRTM'}
        Select one of the DEMs
    parameters : dict
        Dictionary with parameters of the master image
        Output of function get_params_from_xml()
    margin_degrees : float, optional
        Margin in degrees for the dem respect to the image size
    NumThreads : int, optional
        Number of threads to use in the parallel processing steps.

    Returns
    -------
    dem : 3D numpy array
        DEM in the form of a 3D array, where the last dimension representes:
            - (rows, cols,0): Longitude
            - (rows, cols,1): Latitude
            - (rows, cols,2): Height
    dem_xyz : 3D numpy array
        DEM in cartesian coordiantes, the last dimension represents X,Y,Z. respectively.
    dem_posting: float
    dem_limits : dict
        Limits of the DEM.
        It contains the following keys:  {'minlon': ,'maxlon': ,'minlat': ,'maxlat': 0.}
    dang : float

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('get_dem')
    log.info('Get DEM from '+ type_dem)

    if type_dem == 'SRTM':
        #if the teil is not in the path, the SRTMHandler will download the file from internet
        handler = demgen.SRTMHandler(Path(path_dem), download=True)
    elif type_dem == 'TANDEM':
        handler = tandemhandler.tandem_handler(path_dem,type_map='DEM')

    # corners for SRTMHandler in the following order: low left, low right, up left, up right
    # corners from parameters['corners']:  upper right, upper left, lower right ,lower left
    corners = np.empty([1, 4, 2], np.float64)
    lons = np.array([parameters['corners'][3][1], parameters['corners'][2][1], parameters['corners'][1][1],
                     parameters['corners'][0][1]])
    lats = np.array([parameters['corners'][3][0], parameters['corners'][2][0], parameters['corners'][1][0],
                     parameters['corners'][0][0]])

    # add margin following TAXI code
    maxlat = np.max(lats)
    minlat = np.min(lats)
    maxlon = np.max(lons)
    minlon = np.min(lons)

    maxlat += margin_degrees
    minlat -= margin_degrees
    maxlon += margin_degrees
    minlon -= margin_degrees

    # # adding ~10km margin
    meanlat = (minlat + maxlat) * 0.5
    Rearth = 6371000.0  # earth radius
    onedeglon = np.abs(Rearth * np.cos(meanlat / (180 / np.pi)) / (180 / np.pi))  # approx. 1º posting
    onedeglat = np.abs(Rearth / (180 / np.pi))  # approx. 1º posting
    marginlon = 10000 / onedeglon
    marginlat = 10000 / onedeglat
    # (to avoid large matrices in longitude)
    if meanlat > 60: marginlon = marginlat
    maxlat += marginlat
    minlat -= marginlat
    maxlon += marginlon
    minlon -= marginlon

    #add a bit more margin to round the values to avoid shifts in dem generation by tandem handler
    #maxlat = np.float32(np.ceil(maxlat*4.0))/4.0
    #maxlon = np.float32(np.ceil(maxlon*4.0))/4.0
    #minlat = np.float32(np.floor(minlat*4.0))/4.0
    #minlon = np.float32(np.floor(minlon*4.0))/4.0
    maxlat = np.float32(np.ceil(maxlat))
    maxlon = np.float32(np.ceil(maxlon))
    minlat = np.float32(np.floor(minlat))
    minlon = np.float32(np.floor(minlon))

    # Corners for the polygon to get the dem    
    corners[0, :, 0] = np.asarray([minlon, maxlon, minlon, maxlon])
    corners[0, :, 1] = np.asarray([minlat, minlat, maxlat, maxlat])

    handler.read_blocks(corners)
    dem = handler.build_block()

    # Interpolate nans
    log.info('Interpolate nans in the dem ...')
    height_dem = np.copy(np.clip(dem[:, :, 2],0,np.nanmax(dem[:, :, 2])))
    pos_nans = np.isnan(height_dem)
    if np.sum(pos_nans>0):

        #interpolate in the nan positions
        positions_finite = np.where(pos_nans == False)
        positions_nan = np.where(pos_nans==True)
        values_inter = interpolate.griddata(positions_finite,height_dem[pos_nans==False],positions_nan,method='linear')
        #asing interpolate values to nan positions
        height_dem[pos_nans==True] = values_inter

    #clip between 0 and max to avoid negative values due to interpolation
    dem[:,:,2] = np.copy(np.clip(height_dem,0,np.nanmax(height_dem)))
    height_dem = None
    ##check in case there are more nans
    dem[np.isnan(dem)] =  0
    log.info('Interpolate nans in the dem ok')

    #dem[np.isnan(dem)] = 0

    #get limits of the dem
    #lon_min, lon_max, lat_min, lat_max = handler.blockLimits[0]
    
    #get dem posting
    lon_pos,lat_pos = np.mgrid[minlon/np.double(180./np.pi):maxlon/np.double(180./np.pi):dem.shape[0]*1j,minlat/np.double(180./np.pi):maxlat/np.double(180./np.pi):dem.shape[1]*1j]

    lon_pos = lon_pos.ravel()
    lat_pos = lat_pos.ravel()
    latderiv = np.abs(lat_pos-np.roll(lat_pos,1))
    dlat = np.min(latderiv[latderiv!=0])
    londeriv = np.abs(lon_pos-np.roll(lon_pos,1))
    dlon = np.min(londeriv[londeriv!=0])
    dang = np.min([dlat,dlon])
    dem_posting = np.round(dang*(180.0/np.pi)*onedeglon*0.9,0)


    dem_limits = {}
    dem_limits['minlon'] = minlon
    dem_limits['maxlon'] = maxlon
    dem_limits['minlat'] = minlat
    dem_limits['maxlat'] = maxlat
    
    ### transform dem from lon,lat,height to cartesian (x,y,z) coordinates
    lon_pos, lat_pos = np.mgrid[dem_limits['minlon']:dem_limits['maxlon']:dem.shape[0] * 1j,dem_limits['minlat']:dem_limits['maxlat']:dem.shape[1] * 1j]
    
    demllh = np.empty_like(dem)
    demllh[:, :, 0] = np.copy(lon_pos)  # lon
    demllh[:, :, 1] = np.copy(lat_pos)  # lat
    demllh[:, :, 2] = np.copy(dem[:, :, 2])  # height
    
    dem_xyz = geolib.ellip2cart(demllh, num_threads=NumThreads)

    return dem,dem_xyz,dem_posting,dem_limits,dang

def get_offnadir_lookangle(parameters,dem,NumThreads=5):
    """Get off nadir and look angles

    Parameters
    ----------
    parameters : dict
        Dictionary with parameters of the master image
        Output of function get_params_from_xml()
    dem : 3D numpy array
        DEM in the form of a 3D array, where the last dimension representes:
            - (rows, cols,0): Longitude
            - (rows, cols,1): Latitude
            - (rows, cols,2): Height
    Numhreads : int, optional
        Number of threads to use in the parallel processing steps.

    Returns
    -------
    offnadir : float
    lookangle : float

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('get_offnadir_lookangle')
    log.info('Computing offnadir and look angles...')

    # get mean (but using median as in TAXI) height of the dem
    meanheight = np.median(dem[:, :, 2])

    ##get lat lons of the image
    lons = np.array([parameters['corners'][3][1], parameters['corners'][2][1], parameters['corners'][1][1],
                     parameters['corners'][0][1]])
    lats = np.array([parameters['corners'][3][0], parameters['corners'][2][0], parameters['corners'][1][0],
                     parameters['corners'][0][0]])

    # check longitude vaues
    pos = lons > 180
    if np.sum(pos) > 0: lons[pos] = lons[pos] - 360

    # get position of the middel
    posmid = parameters['orbit'][int(parameters['orbit'].shape[0] / 2), 1:4]
    scenemid = np.asarray([np.mean(lons), np.mean(lats), meanheight])
    # transform to cart
    scenemid_aux = np.zeros((1, 1, 3), 'float64')
    scenemid_aux[0, 0, :] = scenemid
    scenemid_cart = geolib.ellip2cart(scenemid_aux,num_threads=NumThreads)

    # compute look angle
    v1 = scenemid_cart - posmid
    v2 = -posmid
    lookangle = np.arccos(np.sum(v1 * v2) / (np.sqrt(np.sum(np.square(v1))) * np.sqrt(np.sum(np.square(v2)))))

    # approx. offnadir angle
    v1 = posmid - scenemid_cart
    v2 = scenemid_cart
    offnadir = np.arccos(np.sum(v1 * v2) / (np.sqrt(np.sum(np.square(v1))) * np.sqrt(np.sum(np.square(v2)))))

    return offnadir, lookangle

def get_params_back_geocoding_dem(parameters, posting,offnadir):
    """Compute auxiliary parameteres for the back-geocoding od the DEM

    Parameters
    ----------
    parameters : dict
        Dictionary with parameters of the master image
        Output of function get_params_from_xml()
    posting : float
        output from get_dem()
    offnadir: float
        output from get_offnadir_lookangle()

    Returns
    -------
    deltat_dem : float
        Sampling in backgeocoded DEM
    rd_dem : float
        Range delay of back-geocoded DEM
    rs_dem : float
        Range sampling of back-geocoded DEM
    t0_dem : float
        Azimuth start time of DEM with margin
    nrg_dem : float
        Range dimensions in pixels
    naz_dem : float
        Azimuth dimensions in pixels

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """


    log = logging.getLogger('get_params_back_geocoding_dem')
    log.info('Computing auxiliary parameters for backgeocoding DEM ...')
    
    #Mean satellite velocity  vs of TAXI
    mean_sat_vel = np.mean(np.sqrt(np.square(parameters['orbit'][:, 4]) + np.square(parameters['orbit'][:, 5]) + np.square(parameters['orbit'][:, 6])))
    # Grpund velocity at mid rag vg of TAXI
    ground_vel_mid_rg = np.square(parameters['effective_vel_mid_rg']) / mean_sat_vel
    # t0_start from TAXI, which is set to 0
    ts_min = 0
    # squint angle is 0, i use the variables instead of 0 to leave the code for future squint adcquisitions
    squint = 0
    azMargin = 0

    
    offnadireff = 0.5 * (2 * np.sin(offnadir))

    ### Range parameters
    dr = constants.c / 2.0 / parameters['rg_sampling_freq']  # slant-range sampling in image
    gdr = dr / np.sin(offnadireff)  # ground-range sampling (aprox)
    ground = parameters['n_rg'] * gdr  # ground range scene extension
    rs_dem = posting * np.sin(offnadireff)  # range sampling of back-geocoded DEM
    margin = 10000.0
    nrg_dem = np.round((ground + margin * 2) / posting * 0.5, 0) * 2  # give a margin of 10km
    incte = margin * np.sin(offnadireff) * 2.0 / constants.c  # 5 km margin at near range
    rd_dem = np.max([0, parameters['rg_delay'] - incte])  # range delay of back-geocoded DEM

    ##azimth parameters
    squintmargin = np.round(np.abs((parameters['rg_delay'] * constants.c / 2.0 + parameters['n_rg'] * dr) * np.tan(squint)) * 0.5, 0)
    dx = ground_vel_mid_rg * (1.0 / parameters['commonPRF'])  # aximuth sampling in focused image (aprox.) ((1.0 / parameters['commonPRF'])  is deltat from TAXI)
    az = parameters['n_az'] * dx  # azimuth extension of image (approx.)
    deltat_dem = posting / ground_vel_mid_rg  # sampling in backgeocoded deM
    naz_dem = np.round((az + 10000 + 2 * azMargin * ground_vel_mid_rg + 2 * squintmargin) / posting * 0.5, 0) * 2.0  # azimuth dimension with margin of 10 km
    t0_dem = ts_min - 5000.0 / ground_vel_mid_rg - squintmargin / ground_vel_mid_rg - azMargin  # azimuth start time of DEM with 5 km margin


    return deltat_dem, rd_dem, rs_dem, t0_dem, nrg_dem, naz_dem

def xyz2rgaz(parameters,dem_xyz,dem_limits,deltat_dem,rd_dem,rs_dem,t0_dem,nrg_dem, naz_dem,is_mono=True,is_bistatic=False,rs_dem_master=0,orbit_master=0,NumThreads=5):
    """Transform from cartesian coordinates xyz to rg,az

    Parameters
    ----------
    parameters : dict
        Dictionary with parameters of the master image
        Output of function get_params_from_xml()
    dem_xyz : 3D numpy array
        DEM in cartesian coordiantes, the last dimension represents X,Y,Z. respectively.
        output from get_dem()
    dem_limits : dict
        Limits of the DEM.
        It contains the following keys:  {'minlon': ,'maxlon': ,'minlat': ,'maxlat': 0.}
        output from get_dem()
    deltat_dem : float
        Sampling in backgeocoded DEM
        Output from get_params_back_geocoding_dem()
    rd_dem : float
        Range delay of back-geocoded DEM
        Output from get_params_back_geocoding_dem()
    rs_dem : float
        Range sampling of back-geocoded DEM
        Output from get_params_back_geocoding_dem()
    t0_dem : float
        Azimuth start time of DEM with margin
        Output from get_params_back_geocoding_dem()
    nrg_dem : float
        Range dimensions in pixels
        Output from get_params_back_geocoding_dem()
    naz_dem : float
        Azimuth dimensions in pixels
        Output from get_params_back_geocoding_dem()
    is_mono : bool, optional
    is_bistatic : bool, optional
    rs_dem_master : float
        Range sampling of back-geocoded DEM
        Output from get_params_back_geocoding_dem()
    orbit_master : 2D numpy array of dimensions (n,7), optional
        Output from interpol_orbits() saved in parameters['orbit_active']
    NumThreads : int, optional
        Number of threads to use in the parallel processing steps.

    Returns
    -------
    rgm: 2D numpy array
        range positions
    azm : 2D numpy array
        Azimuth positions

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('xyz2rgaz')
    log.info('Get range-azimuth matrices ...')


    #### this is the part of the code that in  TAXI is outside the xyz2rgz function
    r_near = parameters['rg_delay'] * constants.c / 2.0
    #r_dem_near = rd_dem * constants.c / 2.0
    
    
    ## check if it is bistatic
    if is_bistatic == False:
    
        rs_local = parameters['spacing_rg']
        rs_dem_local = rs_dem
    else:
    
        r_near = r_near*2.0
        rs_local = 2.0*parameters['spacing_rg']
        #r_dem_near = r_dem_near*2.0
        rs_dem_local = 2.0*rs_dem_master

    ###-------------

    ### ----- From here the function xyz_rgaz_thread of TAXI ------    
    if is_mono ==True:
        r_near = r_near * 2
        rs_local = rs_local * 2

        
        
    dimx = parameters['orbit'].shape[0]
    #prf = parameters['commonPRF']
    t0_start = 0
    orbit_margin = (t0_start - parameters['orbit'][0, 0]) / (1 / parameters['commonPRF'])
    

    # extent orbit
    eastwin = np.round(deltat_dem / (1 / parameters['commonPRF']) * 2, 0)
    if eastwin < 90.0: eastwin = 90.0
    if eastwin < 500:
        fill = 500.0
    else:
        fill = eastwin

    t1 = np.arange(dimx)
    t2 = np.arange(dimx + fill * 2) - fill
    # interpolate time and positions
    func = interpolate.interp1d(t1, np.copy(parameters['orbit'][:, 0]), fill_value="extrapolate")
    ta = func(t2)
    func = interpolate.interp1d(t1, np.copy(parameters['orbit'][:, 1]), fill_value="extrapolate")
    ox = func(t2)
    func = interpolate.interp1d(t1, np.copy(parameters['orbit'][:, 2]), fill_value="extrapolate")
    oy = func(t2)
    func = interpolate.interp1d(t1, np.copy(parameters['orbit'][:, 3]), fill_value="extrapolate")
    oz = func(t2)

    if is_mono==True:
        ox_active = np.copy(ox)
        oy_active = np.copy(oy)
        oz_active = np.copy(oz)
    else:
        # if it is the slave, we use the orbit of the master (active)
        func = interpolate.interp1d(t1, orbit_master[:, 1], fill_value="extrapolate")
        ox_active = func(t2)
        func = interpolate.interp1d(t1, orbit_master[:, 2], fill_value="extrapolate")
        oy_active = func(t2)
        func = interpolate.interp1d(t1, orbit_master[:, 3], fill_value="extrapolate")
        oz_active = func(t2)


    # side vector (for left and right discrimination)
    nadir = -parameters['orbit'][int(dimx * 0.5), 1:4]
    velvec = parameters['orbit'][int(dimx * 0.5), 4:7]
    sidev = np.cross(nadir, velvec)
    sidev = sidev / np.sqrt(np.sum(np.square(sidev)))
    if parameters['look_dir'] == 'left': sidev = -sidev

    # preparation inputs for cart2radar
    Na = int(len(ta) / 2) * 2
    ta = ta[0:Na]
    p_tx = np.zeros((len(ta), 3), dtype='double')
    p_tx[:, 0] = ox[0:Na]
    p_tx[:, 1] = oy[0:Na]
    p_tx[:, 2] = oz[0:Na]

    if is_mono == True:
        p_rx = np.copy(p_tx)
    else:
        p_rx = np.zeros((len(ta), 3), dtype='double')
        p_rx[:, 0] = ox_active[0:Na]
        p_rx[:, 1] = oy_active[0:Na]
        p_rx[:, 2] = oz_active[0:Na]
        
        
    # get range and azimuth matrices
    r0, indx = geolib.cart2radar(dem_xyz.copy(), ta.copy(), p_tx.copy(),p_rx=np.copy(p_rx),bistatic=1, return_time=False,num_threads=NumThreads)

    # create a invalid mask
    invalMask = np.empty_like(r0, dtype='byte')
    invalMask[::] = 1
    invalMask[~np.isnan(r0)] = 0
    invalMask[~np.isnan(indx)] = 0

    # convert to image samples
    azm = indx - (orbit_margin + fill)
    rgm = (r0 - r_near) / rs_local

    midorbit = np.asarray([ox[int(dimx * 0.5)], oy[int(dimx * 0.5)], oz[int(dimx * 0.5)]])

    # checking for antenna direction (make indices on the wrong side negative)
    dotP = (dem_xyz[:, :, 0] - midorbit[0]) * sidev[0] + (dem_xyz[:, :, 1] - midorbit[1]) * sidev[1] + (
                dem_xyz[:, :, 2] - midorbit[2]) * sidev[2]
    sideMask = dotP > 0

    # apply masks
    rgm[invalMask == 1] = np.nan
    azm[invalMask == 1] = np.nan
    sideMask[invalMask == 1] = False

    rgm[sideMask == False] = np.nan
    azm[sideMask == False] = np.nan

    return rgm,azm

def get_dem_height_from_rg_az(rgm,azm,parameters,dem,deltat_dem,rd_dem,rs_dem,t0_dem,nrg_dem, naz_dem):
    """Get the dem height in range/azimuth coordiantes

    Warning: This is the Dem sampling not the SLC sampling

    Parameters
    ----------
    rgm: 2D numpy array
        range positions
        Output from xyz2rgaz()
    azm : 2D numpy array
        Azimuth positions
        Output from xyz2rgaz()
    parameters : dict
        Dictionary with parameters of the master image
        Output of function get_params_from_xml()
    dem : 3D numpy array
        DEM in the form of a 3D array, where the last dimension represents:
            - (rows, cols,0): Longitude
            - (rows, cols,1): Latitude
            - (rows, cols,2): Height
        Output of get_dem()
    deltat_dem : float
        Sampling in backgeocoded DEM
        Output from get_params_back_geocoding_dem()
    rd_dem : float
        Range delay of back-geocoded DEM
        Output from get_params_back_geocoding_dem()
    rs_dem : float
        Range sampling of back-geocoded DEM
        Output from get_params_back_geocoding_dem()
    t0_dem : float
        Azimuth start time of DEM with margin
        Output from get_params_back_geocoding_dem()
    nrg_dem : float
        Range dimensions in pixels
        Output from get_params_back_geocoding_dem()
    naz_dem : float
        Azimuth dimensions in pixels
        Output from get_params_back_geocoding_dem()

    Returns
    -------
    dem_out : 2D numpy array
        dem height in range-azimuth coordinates

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020


    """

    log = logging.getLogger('get_dem_height_from_rg_az')
    log.info('Get dem in range-azimuth coordinates ...')


    #incrd = parameters['rg_delay'] - rd_dem
    #rd_dem = parameters['rg_delay'] - incrd
    ts_min = 0  # t0_start from TAXI, which is set to 0

    #convert azm/rgm matrices to dem positions
    deltat_img = (1.0/parameters['commonPRF'])
    azmdem = (azm*deltat_img+ts_min-t0_dem)/deltat_dem

    rs_img = parameters['spacing_rg']
    rn_img = parameters['rg_delay']*constants.c/2.0
    rn_dem = rd_dem*constants.c/2.0
    rgmdem = (rgm*rs_img+rn_img-rn_dem)/rs_dem
    dem_height = np.copy(dem[:,:,2])

    ##make transpose of azmdem/rgdem to be equal as IDL
    azmdem = np.transpose(azmdem)
    rgmdem = np.transpose(rgmdem)
    dem_height = np.transpose(dem_height)

    # Use grid data to interpolate to the desired positions
    grid_az_out, grid_rg_out = np.mgrid[0:naz_dem:naz_dem * 1j, 0:nrg_dem:nrg_dem * 1j]
    valid_pos = np.isfinite(azmdem)
    azmdem_ravel = azmdem[valid_pos]
    rgmdem_ravel = rgmdem[valid_pos]
    values = dem_height[valid_pos]
    points = np.zeros((len(azmdem_ravel), 2))
    points[:, 0] = azmdem_ravel
    points[:, 1] = rgmdem_ravel
    dem_out = interpolate.griddata(points, values, (grid_az_out, grid_rg_out), method='linear')
    
    return dem_out


def from_dem_dims_to_slc_dims(parameters,dem_radar_coord,nrg_dem,naz_dem,rd_dem,rs_dem,deltat_dem,t0_dem,function_interp='griddata'):
    """Transform image in range/azimuth sampling simensions to the input SLC dimensions

    Convert and image from the sampling used in the DEM to the SLC sampling
    Warning!: Not be confused by the naming of the varaibles. It can be whatever image in radar coordinates with the DEM sampling

    Parameters
    ----------
    parameters : dict
        Dictionary with parameters of the master image
        Output of function get_params_from_xml()
    dem_radar_coord : 2D numpy array
        Image in range-azimuth coordinates with the sampling of the DEM used in the processing
    nrg_dem : float
        Range dimensions in pixels
        Output from get_params_back_geocoding_dem()
    naz_dem : float
        Azimuth dimensions in pixels
        Output from get_params_back_geocoding_dem()
    rd_dem : float
        Range delay of back-geocoded DEM
        Output from get_params_back_geocoding_dem()
    rs_dem : float
        Range sampling of back-geocoded DEM
        Output from get_params_back_geocoding_dem()
    deltat_dem : float
        Sampling in backgeocoded DEM
        Output from get_params_back_geocoding_dem()
    t0_dem : float
        Azimuth start time of DEM with margin
        Output from get_params_back_geocoding_dem()
    function_interp : str
        type of function to interpolate
            - RectBivariateSpline
            - RegularGridInterpolator
            - griddata

    Returns
    -------
    dem_interp_slc : 2D numpy array
        Image in radar coordinates with the same dimensions as the SLC

    """

    log = logging.getLogger('from_dem_dims_to_slc_dims')
    log.info('Interpolate image in DEM dims to SLC image dimensions ...')

    #incrd = parameters['rg_delay'] - rd_dem
    #rd_dem = parameters['rg_delay'] - incrd
    
    t0_img = 0
    deltat_img = (1.0 / parameters['commonPRF'])

    #get rg positions of the slc
    rgvec_dem = np.linspace(start=rd_dem * constants.c / 2.0,
                            stop=(nrg_dem - 1) * rs_dem + (rd_dem * constants.c / 2.0),
                            num=int(nrg_dem))
    rgvec_img = np.linspace(start=parameters['rg_delay'] * constants.c / 2.0,
                            stop=(parameters['n_rg'] - 1) * parameters['spacing_rg'] + (parameters['rg_delay'] * constants.c / 2.0),
                            num=int(parameters['n_rg']))
    f_rgpos = interpolate.interp1d(rgvec_dem, np.arange(nrg_dem))
    rgpos = f_rgpos(rgvec_img)

    #get azimuth positions of the slc
    azvec_dem = np.linspace(start=t0_dem,
                            stop=(naz_dem - 1) * deltat_dem + t0_dem,
                            num=int(naz_dem))
    azvec_img = np.linspace(start=t0_img,
                            stop=(parameters['n_az'] - 1) * deltat_img + t0_img,
                            num=int(parameters['n_az']))

    f_azpos = interpolate.interp1d(azvec_dem, np.arange(naz_dem))
    azpos = f_azpos(azvec_img)

    #interpolation to the desired positoins
    if function_interp == 'RectBivariateSpline':
        f_img = interpolate.RectBivariateSpline(np.arange(naz_dem), np.arange(nrg_dem), dem_radar_coord, kx=1, ky=1)
        dem_interp_slc = f_img(azpos, rgpos)
    elif function_interp == 'RegularGridInterpolator':
        f_img = interpolate.RegularGridInterpolator((np.arange(naz_dem), np.arange(nrg_dem)), dem_radar_coord, method='linear', bounds_error=False)
        points_out = np.meshgrid(azpos, rgpos, indexing='ij')
        points_out_list = np.reshape(points_out, (2, -1), order='C').T
        dem_interp_slc = f_img(points_out_list)
        dem_interp_slc = np.reshape(dem_interp_slc, (len(azpos),len(rgpos)))
    else:
        #in other cases we use griddata
        grid_az, grid_rg = np.mgrid[0:naz_dem:naz_dem * 1j, 0:nrg_dem:nrg_dem * 1j]
        grid_az_out_interp, grid_rg_out_interp = np.meshgrid(azpos, rgpos)
        values = dem_radar_coord.ravel()
        points = np.zeros((len(values), 2))
        points[:, 0] = grid_az.ravel()
        points[:, 1] = grid_rg.ravel()
        dem_interp_slc = interpolate.griddata(points, values, (grid_az_out_interp, grid_rg_out_interp), method='linear')
        #tranpose to have azimth and range
        dem_interp_slc = np.transpose(dem_interp_slc)

    return dem_interp_slc

def get_dem_xyz_flat_earth(dem,dem_limits,dang,NumThreads=5):
    """ Obtain the flat earth dem in cartesian coordinates
    
    Parameters
    ----------
    dem : 3D numpy array
        DEM in the form of a 3D array, where the last dimension represents:
            - (rows, cols,0): Longitude
            - (rows, cols,1): Latitude
            - (rows, cols,2): Height
    dem_limits : dict
        Limits of the DEM.
        It contains the following keys:  {'minlon': ,'maxlon': ,'minlat': ,'maxlat': 0.}
        output from get_dem()
    dang : float
        Output from get_dem()
    NumThreads : int, optional
        Number of threads to use in the parallel processing steps.

    Returns
    -------
    dem_xyz_flat : 3D numpy array:
            - (rows, cols,0): X
            - (rows, cols,1): Y
            - (rows, cols,2): Z
    
    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """


    log = logging.getLogger('get_dem_xyz_flat_earth')
    log.info('Get flat earth DEM  ...')

    meanheight = np.median(dem[:, :, 2])
    Ndec = np.double(1.0)
    #posting_flat = dem_posting * Ndec
    nlon_ = dem.shape[0]
    nlat_ = dem.shape[1]

    lon_axis = np.linspace(start=dem_limits['minlon']* (np.pi / 180.0),
                           stop=(nlon_ - 1) * dang * Ndec + (dem_limits['minlon'] * (np.pi / 180.0)),
                           num=nlon_)

    lat_axis = np.linspace(start=dem_limits['minlat']* (np.pi / 180.0),
                           stop=(nlat_ - 1) * dang * Ndec + (dem_limits['minlat'] * (np.pi / 180.0)),
                           num=nlat_)
    
    #lon_axis2 = np.arange(nlon_)*dang*Ndec + (dem_limits['minlon']* (np.pi / 180.0))

    # transform to degree
    lon_axis = lon_axis * (180.0/np.pi)
    lat_axis = lat_axis * (180.0/np.pi)

    dem_xyz_flat = np.empty_like(dem)
    dem_xyz_flat[::] = np.nan
    aux_axis_height = np.copy(np.repeat(meanheight, nlon_))
    for i_lat, value_i_lat in enumerate(lat_axis):
        demllh = np.zeros((nlon_, 1, 3))
        demllh[:, 0, 0] = np.copy(lon_axis)  # lon
        demllh[:, 0, 1] = np.copy(np.repeat(value_i_lat, nlon_))  # lat
        demllh[:, 0, 2] = aux_axis_height  # height
        dem_xyz_flat_aux = geolib.ellip2cart(demllh.copy(),num_threads=NumThreads)
        dem_xyz_flat[:, i_lat, 0] = dem_xyz_flat_aux[:, 0, 0]
        dem_xyz_flat[:, i_lat, 1] = dem_xyz_flat_aux[:, 0, 1]
        dem_xyz_flat[:, i_lat, 2] = dem_xyz_flat_aux[:, 0, 2]

    return dem_xyz_flat


def compute_slant_phase_flat(parameters,parameters_slave,rgm_flat_master,azm_flat_master,rgm_flat_slave,
                             nrg_dem,naz_dem,rd_dem,rs_dem,deltat_dem,t0_dem):
    """Compute phase of phase flat in slant range geometry

    Parameters
    ----------
    parameters : dict
        Dictionary with parameters of the master image.
        Output of function get_params_from_xml()
    parameters_slave : dict
        Dictionary with parameters of the slave image.
        Output of function get_params_from_xml()
    rgm_flat_master : 2D numpy array
        output from xyz2rgaz
    azm_flat_master : 2D numpy array
        output from xyz2rgaz
    rgm_flat_slave : 2D numpy array
        output from xyz2rgaz
    nrg_dem : float
        Range dimensions in pixels
        Output from get_params_back_geocoding_dem()
    naz_dem : float
        Azimuth dimensions in pixels
        Output from get_params_back_geocoding_dem()
    rd_dem : float
        Range delay of back-geocoded DEM
        output from get_params_back_geocoding_dem()
    rs_dem : float
        Range sampling of back-geocoded DEM
        output from get_params_back_geocoding_dem()
    deltat_dem : float
        Sampling in backgeocoded DEM
        output from get_params_back_geocoding_dem()
    t0_dem: float
        Azimuth start time of DEM with margin
        output from get_params_back_geocoding_dem()

    Returns
    -------
    slantphaseflat : 2D numpy array
        Phase flat in slant range geometry

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('compute_slant_phase_flat')
    log.info('Compute slant phase flat ...')


    ts_min = 0  # t0_start from TAXI, which is set to 0

    # convert azm/rgm matrices to dem positions
    deltat_img = (1.0 / parameters['commonPRF'])
    azmdem = (azm_flat_master * deltat_img + ts_min - t0_dem) / deltat_dem

    rs_img_master = parameters['spacing_rg']
    rn_img_master = parameters['rg_delay'] * constants.c / 2.0
    rn_dem = rd_dem * constants.c / 2.0
    rgmdem = (rgm_flat_master * rs_img_master + rn_img_master - rn_dem) / rs_dem

    rn_img_slave = parameters_slave['rg_delay'] * constants.c / 2.0

    # ##make transpose of azmdem/rgdem to be equal as IDL ????
    azmdem = np.transpose(azmdem)
    rgmdem = np.transpose(rgmdem)
    rgm_flat_slave = np.transpose(rgm_flat_slave)
    rgm_flat_master = np.transpose(rgm_flat_master)


    if parameters['spacing_rg'] == parameters_slave['spacing_rg']:
        values_matrix = rgm_flat_master - rgm_flat_slave
        # Use grid data to interpolate to the desired positions
        grid_az_out, grid_rg_out = np.mgrid[0:naz_dem:naz_dem * 1j, 0:nrg_dem:nrg_dem * 1j]
        valid_pos = np.isfinite(azmdem)
        azmdem_ravel = azmdem[valid_pos]
        rgmdem_ravel = rgmdem[valid_pos]

        values = values_matrix[valid_pos]
        points = np.zeros((len(azmdem_ravel), 2))
        points[:, 0] = azmdem_ravel
        points[:, 1] = rgmdem_ravel
        data = interpolate.griddata(points, values, (grid_az_out, grid_rg_out), method='linear')
        slantphaseflat = -4 * np.pi / (constants.c / parameters['f0']) * (rn_img_master - rn_img_slave + data * rs_img_master)
    else:
        # In case the row spacing is different for master and slave

        # Use grid data to interpolate to the desired positions
        grid_az_out, grid_rg_out = np.mgrid[0:naz_dem:naz_dem * 1j, 0:nrg_dem:nrg_dem * 1j]
        valid_pos = np.isfinite(azmdem)
        azmdem_ravel = azmdem[valid_pos]
        rgmdem_ravel = rgmdem[valid_pos]

        ## For master
        values_master = rgm_flat_master[valid_pos]
        points = np.zeros((len(azmdem_ravel), 2))
        points[:, 0] = azmdem_ravel
        points[:, 1] = rgmdem_ravel
        datam = interpolate.griddata(points, values_master, (grid_az_out, grid_rg_out), method='linear')

        ## For slave
        values_master = rgm_flat_slave[valid_pos]
        points = np.zeros((len(azmdem_ravel), 2))
        points[:, 0] = azmdem_ravel
        points[:, 1] = rgmdem_ravel
        datas = interpolate.griddata(points, values_master, (grid_az_out, grid_rg_out), method='linear')

        slantphaseflat = -4 * np.pi / (constants.c / parameters['f0']) * (rn_img_master - rn_img_slave + datam*rs_img_master-datas*parameters_slave['spacing_rg'])

    return slantphaseflat


def rgz2xyz(parameters,rgm,azm,dem_xyz,deltat_dem,rd_dem,rs_dem,t0_dem,nrg_dem, naz_dem):
    """Transform from slant/range to cartesian coordinate

    Parameters
    ----------
    parameters : dict
        Dictionary with parameters of the master image.
        Output of function get_params_from_xml()
    rgm : 2D numpy array
        range positions
        Output from xyz2rgaz()
    azm : 2D numpy array
        Azimuth positions
        Output from xyz2rgaz()
    dem_xyz : 3D numpy array
        dem in cartesian coordaintes
        ouput from get_dem()
    deltat_dem : float
        Sampling in backgeocoded DEM
        output from get_params_back_geocoding_dem()
    rd_dem : float
        Range delay of back-geocoded DEM
        output from get_params_back_geocoding_dem()
    rs_dem : float
        Range sampling of back-geocoded DEM
        output from get_params_back_geocoding_dem()
    t0_dem: float
        Azimuth start time of DEM with margin
        output from get_params_back_geocoding_dem()
    nrg_dem : float
        Range dimensions in pixels
        Output from get_params_back_geocoding_dem()
    naz_dem : float
        Azimuth dimensions in pixels
        Output from get_params_back_geocoding_dem()

    Returns
    -------
    dem_xyz_slr : 3d numpy array
            - (rows, cols,0): X
            - (rows, cols,1): Y
            - (rows, cols,2): Z

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('rgz2xyz')
    log.info('Get XYZ coordiantes of DEM in slant-range geometry ...')


    ts_min = 0  # t0_start from TAXI, which is set to 0

    #convert azm/rgm matrices to dem positions
    deltat_img = (1.0/parameters['commonPRF'])
    azmdem = (azm*deltat_img+ts_min-t0_dem)/deltat_dem

    rs_img = parameters['spacing_rg']
    rn_img = parameters['rg_delay']*constants.c/2.0
    rn_dem = rd_dem*constants.c/2.0
    rgmdem = (rgm*rs_img+rn_img-rn_dem)/rs_dem

    ##make transpose of azmdem/rgdem to be equal as IDL
    azmdem = np.transpose(azmdem)
    rgmdem = np.transpose(rgmdem)

    grid_az_out, grid_rg_out = np.mgrid[0:naz_dem:naz_dem * 1j, 0:nrg_dem:nrg_dem * 1j]
    valid_pos = np.isfinite(azmdem)
    azmdem_ravel = azmdem[valid_pos]
    rgmdem_ravel = rgmdem[valid_pos]
    points = np.zeros((len(azmdem_ravel), 2))
    points[:, 0] = azmdem_ravel
    points[:, 1] = rgmdem_ravel

    #output matrix
    dem_xyz_slr = np.zeros((int(naz_dem), int(nrg_dem), 3))
    
    #interpolate x,y,z separatelly
    for i_position in range(3):
        dem_xyz_aux = np.transpose(np.copy(dem_xyz[:,:,i_position]))
        values = dem_xyz_aux[valid_pos]
        dem_xyz_slr[:,:,i_position] = interpolate.griddata(points, values, (grid_az_out, grid_rg_out), method='linear')
        

    return dem_xyz_slr


def get_offsets(rgm,azm,rgs,azs,parameters,parameters_slave,dem,deltat_dem,rd_dem,rs_dem,t0_dem,nrg_dem, naz_dem):
    """Computation of azimuth offsets

    Parameters
    ----------
    rgm : 2D numpy array
        range positions
        Output from xyz2rgaz()
    azm : 2D numpy array
        Azimuth positions
        Output from xyz2rgaz()
    rgs : 2D numpy array
        output from xyz2rgaz
    azs : 2D numpy array
        output from xyz2rgaz
    parameters : dict
        Dictionary with parameters of the master image.
        Output of function get_params_from_xml()
    parameters : dict
        Dictionary with parameters of the slave image.
        Output of function get_params_from_xml()
    dem : 3D numpy array
        DEM in the form of a 3D array, where the last dimension represents:
            - (rows, cols,0): Longitude
            - (rows, cols,1): Latitude
            - (rows, cols,2): Height
    deltat_dem : float
        Sampling in backgeocoded DEM
        output from get_params_back_geocoding_dem()
    rd_dem : float
        Range delay of back-geocoded DEM
        output from get_params_back_geocoding_dem()
    rs_dem : float
        Range sampling of back-geocoded DEM
        output from get_params_back_geocoding_dem()
    t0_dem: float
        Azimuth start time of DEM with margin
        output from get_params_back_geocoding_dem()
    nrg_dem : float
        Range dimensions in pixels
        Output from get_params_back_geocoding_dem()
    naz_dem : float
        Azimuth dimensions in pixels
        Output from get_params_back_geocoding_dem()

    Returns
    -------
    az_offset : 2D numpy array
        Azimuth offsets
    rg_offset : 2D numpy array
        Range offsets
    synth_phase : 2D numpy array
        Synthetic phase in slant range geometry

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020
           January 2021 - Added azimuth offsets and synthetic phase

    """

    log = logging.getLogger('get_offsets')
    log.info('Get azimuth offsets ...')


    # incrd = parameters['rg_delay'] - rd_dem
    # rd_dem = parameters['rg_delay'] - incrd
    ts_min = 0  # t0_start from TAXI, which is set to 0
    
    # convert azm/rgm matrices to dem positions
    deltat_img = (1.0 / parameters['commonPRF'])
    azmdem = (azm * deltat_img + ts_min - t0_dem) / deltat_dem
    
    rs_img = parameters['spacing_rg']
    rs_img_slave = parameters_slave['spacing_rg']
    rn_img = parameters['rg_delay'] * constants.c / 2.0
    rn_img_slave = parameters_slave['rg_delay'] * constants.c / 2.0
    rn_dem = rd_dem * constants.c / 2.0
    rgmdem = (rgm * rs_img + rn_img - rn_dem) / rs_dem
    
    ##make transpose of azmdem/rgdem to be equal as IDL
    azmdem = np.transpose(azmdem)
    rgmdem = np.transpose(rgmdem)
    azm = np.transpose(azm)
    azs = np.transpose(azs)

    azdif = np.copy(azm-azs)
    valid_pos = np.isfinite(azm) * np.isfinite(azs)
    
    ## 1- get offser az

    # Use grid data to interpolate to the desired positions
    grid_az_out, grid_rg_out = np.mgrid[0:naz_dem:naz_dem * 1j, 0:nrg_dem:nrg_dem * 1j]
    azmdem_ravel = azmdem[valid_pos]
    rgmdem_ravel = rgmdem[valid_pos]
    values = azdif[valid_pos]
    points = np.zeros((len(azmdem_ravel), 2))
    points[:, 0] = azmdem_ravel
    points[:, 1] = rgmdem_ravel
    az_offset = interpolate.griddata(points, values, (grid_az_out, grid_rg_out), method='linear')

    ## 2- get offset in range and the synthetic phase in slant-range plane
    rgm = np.transpose(rgm)
    rgs = np.transpose(rgs)
    if parameters['spacing_rg'] == parameters_slave['spacing_rg']:
        rgdif = np.copy(rgm-rgs)
        valid_pos = np.isfinite(rgm) * np.isfinite(rgs)
        values = rgdif[valid_pos]
        rg_offset = interpolate.griddata(points, values, (grid_az_out, grid_rg_out), method='linear')
        synth_phase = (-4*np.pi/(constants.c / parameters['f0']))*(rn_img-rn_img_slave+rg_offset*rs_img)
    else:
        valid_pos = np.isfinite(rgm) * np.isfinite(rgm)
        values= rgm[valid_pos]
        rgm_offset = interpolate.griddata(points, values, (grid_az_out, grid_rg_out), method='linear')
        values= rgs[valid_pos]
        rgs_offset = interpolate.griddata(points, values, (grid_az_out, grid_rg_out), method='linear')

        rg_offset = rgm_offset-rgs_offset
        synth_phase = (-4 * np.pi / (constants.c / parameters['f0'])) * (rn_img - rn_img_slave+ rgm_offset*rs_img - rgs_offset*rs_img_slave)



    return az_offset,rg_offset,synth_phase


def get_baselines(parameters,parameters_slave,dem_xyz_slr,az_offset,deltat_dem,t0_dem):
    """Get baseline parameters

    Parameters
    ----------
    parameters : dict
        Dictionary with parameters of the master image.
        Output of function get_params_from_xml()
    parameters_slave : dict
        Dictionary with parameters of the slave image.
        Output of function get_params_from_xml()
    dem_xyz_slr : 3D numpy array
        Cartesian coordinates of the dem in slant range geometry
        Ouput of rgz2xyz()
    az_offset : 2D numpy array
        Azimuth offsets
        Ouput from get_offsets()
    deltat_dem : float
        Sampling in backgeocoded DEM
        output from get_params_back_geocoding_dem()
    t0_dem: float
        Azimuth start time of DEM with margin
        output from get_params_back_geocoding_dem()

    Returns
    -------
    baseline : 2D numpy array
        Baseline
    bpar : 2D numpy array
        Parallel baseline
    bperp : 2D numpy array
        Perpendicular baseline
    kz : 2D numpy array
        Vertical wavenumber
    thetainc : 2D numpy array
        Inicidence angle

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """


    log = logging.getLogger('get_baselines')
    log.info('Get products: kz, perpendicual baselines,incidence angle ...')

    # inputs
    orbitm = parameters['orbit']
    orbits = 0.5 * (parameters_slave['orbit'] + parameters['orbit_active'])
    x = dem_xyz_slr[:, :, 0]
    y = dem_xyz_slr[:, :, 1]
    z = dem_xyz_slr[:, :, 2]
    azoffm = 0
    azm = 0
    azs = 0
    deltat = np.copy(deltat_dem)
    deltatm = 1.0 / parameters['commonPRF']
    # t0_dem
    t0_img = 0
    t0_orbit = parameters['orbit'][0, 0]
    offset_img = np.round(-parameters_slave['orbit'][0, 0] / (1.0 / parameters_slave['commonPRF']), 0)
    azoff = np.copy(az_offset)

    nrows = x.shape[0]
    ncols = x.shape[1]

    # outputs
    kz = np.zeros((nrows, ncols))
    thetainc = np.zeros((nrows, ncols))
    bperp = np.zeros((nrows, ncols))
    bpar = np.zeros((nrows, ncols))
    baseline = np.zeros((nrows, ncols))

    offset_img_master = round((t0_img - t0_orbit) / deltatm, 0)

    for i_row in range(nrows):
        # azimuth shift pixels
        Daz = azoff[i_row, :]
        mpos = (i_row * deltat - (t0_img - t0_dem)) / deltatm  # master position (at full PRF)
        az = np.clip(mpos - Daz + offset_img, 0, orbits.shape[0] - 2)
        azm = np.clip(mpos + offset_img_master, 0, orbitm.shape[0] - 2)

        # linear interpolation between adjancent positions
        az_int = np.copy(np.asarray(az, np.int64))
        S2a = orbits[az_int, 1:4]
        S2b = orbits[az_int + 1, 1:4]
        az_aux2 = np.reshape(np.repeat(az - az_int, 3), (ncols, 3))
        S2 = S2a + az_aux2 * (S2a - S2b)

        azm = np.zeros(ncols) + azm
        azm_int = np.asarray(azm, np.int64)
        S1a = orbitm[azm_int, 1:4]
        S1b = orbitm[azm_int + 1, 1:4]
        az_aux1 = np.reshape(np.repeat(azm - azm_int, 3), (ncols, 3))
        S1 = S1a + az_aux1 * (S1a - S1b)

        B = S2 - S1

        # baseline plane

        # velocity vector (along
        v = orbitm[azm_int, 4:7]
        v_norm = np.reshape(np.repeat(np.squeeze(np.linalg.norm(v, axis=1, keepdims=True)), 3), (ncols, 3))
        v = v / v_norm

        # Orthogonal to the c laying in the plane of V and B
        Bmod2 = np.reshape(np.repeat(np.sum(np.square(B), 1), 3), (ncols, 3))  # square of baselines
        dotvB2 = np.reshape(np.repeat(np.square(np.sum(v * B, 1)), 3), (ncols, 3))  # dot product between v and B
        aux1 = np.reshape(np.repeat(np.sum(v * B, 1), 3), (ncols, 3))
        a = (B - aux1 * v) / np.sqrt(Bmod2 - dotvB2)

        # orthogonal to the other two
        aperp = np.cross(a, v)

        # target orbit vector
        P = np.zeros((ncols, 3), 'double')
        P[:, 0] = np.copy(x[i_row, :] - S1[:, 0])
        P[:, 1] = np.copy(y[i_row, :] - S1[:, 1])
        P[:, 2] = np.copy(z[i_row, :] - S1[:, 2])

        # vector on baseline plane
        cross_P_aperp = np.cross(P, aperp)
        norm_aperp = np.reshape(np.repeat(np.squeeze(np.linalg.norm(aperp, axis=1, keepdims=True)), 3), (ncols, 3))
        cross_P_aperp_norm = cross_P_aperp / norm_aperp
        Pav = np.cross(aperp, cross_P_aperp_norm) / norm_aperp

        # vector perpendicular to baseline plane
        Ppv = np.reshape(np.repeat(np.sum(P * aperp, 1), 3), (ncols, 3)) * aperp / np.reshape(
            np.repeat(np.sum(np.square(aperp), 1), 3), (ncols, 3))

        # pararllel and perpendicular baselines
        Pa = np.sum(Pav * a, 1)
        Pp = np.sum(Ppv * aperp, 1)

        # vector
        BparallelV = (np.reshape(np.repeat(Pa, 3), (ncols, 3)) * a) + (np.reshape(np.repeat(Pp, 3), (ncols, 3)) * aperp)
        # Normalization
        BparallelV = BparallelV / np.reshape(
            np.repeat(np.squeeze(np.linalg.norm(BparallelV, axis=1, keepdims=True)), 3), (ncols, 3))

        Ba = np.sum(B * a, 1)
        Bparallel = Ba * np.sum(a * BparallelV, 1)
        bperpv = np.cross(BparallelV, v)
        Bperpend = Ba * np.sum(a * bperpv, 1)

        # assign to putput matrices
        baseline[i_row, :] = np.sqrt(np.sum(np.square(B), 1))
        bpar[i_row, :] = Bparallel
        bperp[i_row, :] = Bperpend

        # compute Kz and incidence angle

        # orbit-target vector
        Vec1 = np.zeros((ncols, 3), 'double')
        Vec1[:, 0] = np.copy(S1[:, 0] - x[i_row, :])
        Vec1[:, 1] = np.copy(S1[:, 1] - y[i_row, :])
        Vec1[:, 2] = np.copy(S1[:, 2] - z[i_row, :])

        # Target position vector
        Vec2 = np.zeros((ncols, 3), 'double')
        Vec2[:, 0] = np.copy(x[i_row, :])
        Vec2[:, 1] = np.copy(y[i_row, :])
        Vec2[:, 2] = np.copy(z[i_row, :])

        # incidence angle
        thetainc[i_row, :] = np.arccos(
            np.sum(Vec1 * Vec2, 1) / (np.sqrt(np.sum(np.square(Vec1), 1)) * np.sqrt(np.sum(np.square(Vec2), 1))))

        # kz
        rm = np.sqrt(np.sum(np.square(Vec1), 1))
        kz[i_row, :] = -4 * np.pi / (constants.c / parameters['f0']) * bperp[i_row, :] / (rm * np.sin(thetainc[i_row, :]))

    # remove invalid positions
    pos_invalid = np.isnan(x)

    kz[pos_invalid] = np.nan
    thetainc[pos_invalid] = np.nan
    bperp[pos_invalid] = np.nan
    bpar[pos_invalid] = np.nan
    baseline[pos_invalid] = np.nan

    return baseline,bpar,bperp,kz,thetainc 


def add_same_offset_images(parameters,parameters_slave,rd_dem,rd_dem_slave,t0_dem):
    """Get offsets of the dem for master and slave

    Parameters
    ----------
    parameters : dict
        Dictionary with parameters of the master image.
        Output of function get_params_from_xml()
    parameters_slave : dict
        Dictionary with parameters of the slave image.
        Output of function get_params_from_xml()
    rd_dem : float
         Range sampling of back-geocoded DEM for the master
         Output from get_params_back_geocoding_dem()
    rd_dem_slave: float
        Range sampling of back-geocoded DEM for the slave
        Output from get_params_back_geocoding_dem()
    t0_dem
        Azimuth start time of DEM with margin
        Output from get_params_back_geocoding_dem()

    Returns
    -------
    rd_dem : float
    rd_dem_slave : float
    t0_dem : float
    t0_dem_slave : float

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('add_same_offset_images')
    log.info('Add same offset to the images ...')

    
    incrd = parameters['rg_delay'] - rd_dem
    rd_dem = parameters['rg_delay'] - incrd
    rd_dem_slave = parameters_slave['rg_delay'] - incrd
    ts_min = 0  # t0_start from TAXI, which is set to 0
    inct = ts_min - t0_dem
    t0_dem = ts_min - inct
    t0_dem_slave = ts_min - inct

    return rd_dem,rd_dem_slave,t0_dem,t0_dem_slave


def blocks_griddata_2d(points, values, xi, method='linear', fill_value=np.nan,rescale=False, cuts='auto', extra_space=0.25):
    """Griddata by bloks

    It computes the griddata from scipy by blocks to avoid memory problems when the image is too big

    Parameters
    ----------
    points
    values
    xi
    method
    fill_value
    rescale
    cuts
    extra_space

    Returns
    -------
    out
        Interpolated image

    Notes
    -------
    Author : Alberto Alonso-Gonzalez

    """

    log = logging.getLogger('blocks_griddata_2d')
    log.info('interpolate.griddata by blocks ...')

    if isinstance(points, tuple):
        log.info("Input values as tuple, converting to numpy array")
        assert (len(points) == 2)
        points = np.asarray([points[0], points[1]]).T

    if isinstance(cuts, str):
        if cuts.capitalize() == 'Auto':
            # Try to separate into blocks of approx 2 million points
            nc1 = np.max([np.int(np.round(np.sqrt(points.shape[0] / 2e6))), 1])
            nc2 = np.max([np.int(np.round(points.shape[0] / nc1 / 2e6)), 1])
            cuts = [nc1, nc2]
        else:
            raise ValueError("invalid argument string for cuts param: '{}'".format(cuts))

    extra_space = np.float(extra_space)
    assert (extra_space >= 0.0)
    ndims = points.shape[1]
    assert (ndims == 2)
    log.info("Points shape: {}".format(points.shape))
    log.info("Using {} blocks.".format(cuts))
    mins = np.asarray([np.min(xi[0].ravel()), np.min(xi[1].ravel())])
    maxs = np.asarray([np.max(xi[0].ravel()), np.max(xi[1].ravel())])
    out_shape = xi[0].shape
    out = np.empty(out_shape, dtype=values.dtype)
    out.fill(fill_value)
    widths = maxs - mins
    for i in range(cuts[0]):
        min1 = mins[0] + (np.float(i) - extra_space) * widths[0] / cuts[0]
        max1 = mins[0] + (i + 1.0 + extra_space) * widths[0] / cuts[0]
        minpo1 = mins[0] + np.float(i) * widths[0] / cuts[0]
        maxpo1 = mins[0] + (i + 1.0) * widths[0] / cuts[0]
        for j in range(cuts[1]):
            min2 = mins[1] + (j - extra_space) * widths[1] / cuts[1]
            max2 = mins[1] + (j + 1.0 + extra_space) * widths[1] / cuts[1]
            minpo2 = mins[1] + np.float(j) * widths[1] / cuts[1]
            maxpo2 = mins[1] + (j + 1.0) * widths[1] / cuts[1]
            #log.info("Values outer: ({}-{}, {}-{})".format(min1, max1, min2, max2))
            #log.info("Values inner: ({}-{}, {}-{})".format(minpo1, maxpo1, minpo2, maxpo2))
            vpoints = np.where(
                (points[:, 0] <= max1) & (points[:, 0] >= min1) & (points[:, 1] <= max2) & (points[:, 1] >= min2))
            if (vpoints[0].size > 2):
                vxi = np.where((xi[0] <= maxpo1) & (xi[0] >= minpo1) & (xi[1] <= maxpo2) & (xi[1] >= minpo2))
                out[vxi] = interpolate.griddata(points[vpoints[0], :], values[vpoints[0]], (xi[0][vxi], xi[1][vxi]), method=method,
                                    fill_value=fill_value, rescale=rescale)
    return out


def geocoding_radar_image(image_radar_coord,parameters,dem,NumThreads=5,margin=0.05,pixels_spacing=10,pixels_border_to_remove=50):
    """Geocoding radar image

    It uses the newtonbackgeo lib

    Parameters
    ----------
    image_radar_coord: 2D numpy array
        Image in radar coordaintes with SLC dimensions or a re-scaled version of it.
        Warning: A crop of the image is not accepted.
    parameters : dict
        Dictionary with parameters of the master image.
        Output of function get_params_from_xml()
        WARNING!: It must include the interpolated orbits generated in interpol_orbits()
    dem : 3D numpy array
        DEM in the form of a 3D array, where the last dimension represents:
            - (rows, cols,0): Longitude
            - (rows, cols,1): Latitude
            - (rows, cols,2): Height
    NumThreads : int, optional
        Number of threads to use in the parallel processing steps.
    margin_degrees : float, optional
        Margin in degrees or meters (repect to the limits of the radr image) for the resulting image
    pixels_spacing : float
        Desired pixels spacing for the resulting image
    pixels_border_to_remove : int
         Number of pixels to remove in each border after geocoding
         This is done to remove the wrong pixels in the borders due to errors
           in the interpolation procedure during the geocoding


    Returns
    -------
    image_geo_coord: 2D numpy array
        Image in lat lot coordinates
    row_axis_coord 1D numpy array
        Latitude values for the columns of image_geo_coord
    col_axis_coord: 1D numpy array
        Longitude values for the rows of image_geo_coord

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('geocoding_radar_image')
    log.info('Geocoding a radar image ...')

    #dem for the geocoding
    dem_llh = np.copy(dem)
    dem = None
    dem_xyz = geolib.ellip2cart(dem_llh, num_threads=NumThreads)

    newton = NewtonBackgeocoder(dem_llh, dem_xyz, num_threads=NumThreads)
    dem_llh = None
    dem_xyz = None

    #satellite positoins
    p_sat = np.zeros((int(parameters['n_az']), 3), dtype='double')
    p_sat[:, 0] = np.copy(parameters['orbit_slc'][:, 1])
    p_sat[:, 1] = np.copy(parameters['orbit_slc'][:, 2])
    p_sat[:, 2] = np.copy(parameters['orbit_slc'][:, 3])

    #satellite velocity
    v_sat = np.zeros((int(parameters['n_az']), 3), dtype='double')
    v_sat[:, 0] = np.copy(parameters['orbit_slc'][:, 4])
    v_sat[:, 1] = np.copy(parameters['orbit_slc'][:, 5])
    v_sat[:, 2] = np.copy(parameters['orbit_slc'][:, 6])

    #range to each pixel
    r0 = np.copy(parameters['range_vec'])

    #Make the backgeocoding
    log.info('Get irregular grid lon/lat/height <--> range/azimuth ...')

    ##reduce image
    n_az, n_rg = image_radar_coord.shape

    # Check if the image is re-scales respect to the original slc dimensions
    factor_rescale_rg = n_rg / parameters['n_rg']
    factor_rescale_az = n_az / parameters['n_az']

    #Reduce orbit parameters by the correspongin size
    p_sat = ndimage.zoom(p_sat, (factor_rescale_az, 1), order=0, mode='nearest')
    v_sat = ndimage.zoom(v_sat, (factor_rescale_az, 1), order=0, mode='nearest')
    r0 = ndimage.zoom(r0, (factor_rescale_rg), order=0, mode='nearest')

    #output matrices
    grid_xyz = np.zeros((n_az,n_rg, 3), dtype='double')
    err = np.zeros((n_az,n_rg), dtype='double')
    #number of azimuth to be computed at each call to the backgeocoded method. By default i add 500,
    #   if it is too big then it take much more time than doing it in smaller blocks
    n_azs_for_geocod = 1000
    for i_az in range(0,n_az,n_azs_for_geocod):
        #get the pixels of the block
        if i_az == n_az - np.mod(n_az,n_azs_for_geocod):
            pixel_end = n_az
        else:
            pixel_end = i_az+n_azs_for_geocod
        #make the geocoding of the corresponding block
        grid_xyz_aux, err_aux = newton.backgeocode(p_sat[i_az:pixel_end,:], v_sat[i_az:pixel_end,:], r0)
        grid_xyz[i_az:pixel_end, :, :] = np.copy(grid_xyz_aux)
        err[i_az:pixel_end, :] = np.copy(err_aux)

    p_sat = None
    v_sat = None
    r0 = None
    err = None

    log.info('Get regular grid for image in lat/lon coordiantes ...')

    # transform to lon lat
    grid_llh = geolib.cart2ellip(grid_xyz.copy(), num_threads=NumThreads)
    #
    grid_xyz = None

    # check size
    assert (grid_llh.shape[0] == n_az)
    assert (grid_llh.shape[1] == n_rg)

    # get coorners and add some margin
    min_lon = np.min(grid_llh[:, :, 0]) - margin
    max_lon = np.max(grid_llh[:, :, 0]) + margin
    min_lat = np.min(grid_llh[:, :, 1]) - margin
    max_lat = np.max(grid_llh[:, :, 1]) + margin

    # Seelct the number of lats and lons depending on the pixel spacing provided as an input

    ##to get distance between latlon points using pyproj
    # geod = pyproj.Geod(ellps='WGS84')
    # ang1,ang2,dist_cols = geod.inv(min_lon,min_lat,max_lon,min_lat)
    # ang1, ang2, dist_rows = geod.inv(min_lon, min_lat, min_lon, max_lat)

    # get distance for each dimension using ECEF coordinates
    corners_llh = np.zeros((2, 2, 3))
    corners_llh[0, 0, 0] = min_lon  # lon
    corners_llh[0, 0, 1] = min_lat  # lat
    corners_llh[0, 1, 0] = min_lon
    corners_llh[0, 1, 1] = max_lat
    corners_llh[1, 0, 0] = max_lon
    corners_llh[1, 0, 1] = min_lat
    corners_llh[1, 1, 0] = max_lon
    corners_llh[1, 1, 1] = max_lat
    corners_xyz = geolib.ellip2cart(corners_llh.copy(), num_threads=NumThreads)
    dist_rows = np.sqrt((np.square(corners_xyz[0, 0, 0] - corners_xyz[0, 1, 0])) +
                        (np.square(corners_xyz[0, 0, 1] - corners_xyz[0, 1, 1])) +
                        (np.square(corners_xyz[0, 0, 2] - corners_xyz[0, 1, 2])))
    dist_cols = np.sqrt((np.square(corners_xyz[0, 0, 0] - corners_xyz[1, 0, 0])) +
                        (np.square(corners_xyz[0, 0, 1] - corners_xyz[1, 0, 1])) +
                        (np.square(corners_xyz[0, 0, 2] - corners_xyz[1, 0, 2])))

    # define the numer of lons (rows) and lats (cols) depending in the pixels_spacing
    n_rows = int(np.round(dist_rows / pixels_spacing, 0))
    n_cols = int(np.round(dist_cols / pixels_spacing, 0))


    # ##interpolation to a regular grid

    #grid output image
    grid_x, grid_y = np.mgrid[min_lon:max_lon:n_cols * 1j, min_lat:max_lat:n_rows * 1j]
    # get valules and points for grid data
    valid_pos = np.isfinite(grid_llh[:, :, 0]) * np.isfinite(grid_llh[:, :, 1])
    values = image_radar_coord[valid_pos]
    points = np.zeros((len(values), 2))
    points[:, 0] = grid_llh[valid_pos,0]
    points[:, 1] = grid_llh[valid_pos,1]
    grid_llh = None

    #genreate axis of the image in geografical coordinates
    col_axis_coord = np.linspace(min_lon,max_lon,n_cols)
    row_axis_coord = np.linspace(min_lat,max_lat,n_rows)



    #check the number of input poitns of grid data
    if len(values) < 20e6:
        #then we use normal griddata, asuming that we can do it
        try:
            image_geo_coord = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

        except:
            log.error('Error in grid data to much points for interpolation')

    else:
        #use grid data by blocks as the number of points is too big
        image_geo_coord = blocks_griddata_2d(points, values, (grid_x, grid_y), method='linear')


    # Remove borders of the image, due to potential errors in the interpolation doen in the processing of each individual image
    # 1- Get a mask of the finite points
    pos_finite_numbers = np.isfinite(image_geo_coord)
    # using a uniform filter we decrease the area of valid pixels rom the border
    mask_borders = ndimage.uniform_filter(np.asarray(pos_finite_numbers, 'float'), pixels_border_to_remove)
    mask_borders[mask_borders < 0.95] = 0
    # where the mask is 0 we consider as invalid
    image_geo_coord[mask_borders == 0] = np.nan



    return image_geo_coord,row_axis_coord,col_axis_coord


def compute_noise_decorrelation(resolution,parameters,slc,thetainc,rg_slope):
    """Compute noise decorrelation

    Parameters
    ----------
    resolution : float
        Desired resolution.
    parameters : dict
        Dictionary with parameters of the master image.
        Output of function get_params_from_xml()
    slc : 2d numpy array
        SLC image
    thetainc : 2D numpy array
        Inicidence angle
        Output from get_baselines()
    rg_slope: 2D numpy array
        Slope im range direction
        Output from compute_corrected_kz_and_slope_dem()

    Returns
    -------
    gama_SNR : 2D numpy array

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('compute_noise_decorrelation')
    log.info('Computing noise decorrelation step 1 of 3 ...')


    ml_rg = np.int(np.round(resolution / ((parameters['groundNear'] + parameters['groundFar']) / 2.)))
    ml_az = np.int(np.round(resolution / parameters['projectedSpacingAzimuth']))

    delta_tau = parameters['validity_range_max'][0]-parameters['validity_range_min'][0]
    dtau = delta_tau / parameters['n_rg']
    tau = np.linspace(start=parameters['validity_range_min'][0],stop=int(parameters['n_rg']-1)*dtau+parameters['validity_range_min'][0], num=int(parameters['n_rg']))

    noise_polynomials = np.zeros((int(parameters['n_rg']),int(parameters['n_noise_records'])))


    for i_noise in range(int(parameters['n_noise_records'])):
        for i_order_poly in range(int(parameters['noise_polynomial_degree'])+1):
            noise_polynomials[:,i_noise] = noise_polynomials[:,i_noise] + parameters['noise_coef'][i_noise,i_order_poly] * (np.power(tau-parameters['reference_point'][i_noise],i_order_poly))


    noise_eq_beta_nought = np.multiply(noise_polynomials,parameters['calFactor'])

    #enlarge noise polynomial to full azimuth size
    #noise_eq_beta_nought = ndimage.zoom(noise_eq_beta_nought,(1,parameters['n_az']/noise_eq_beta_nought.shape[1]),order=1)
    #noise_eq_beta_nought = np.multiply(noise_eq_beta_nought.transpose(),np.sin(thetainc))

    noise_eq_beta_nought = ndimage.zoom(noise_eq_beta_nought,(1,parameters['n_az']/noise_eq_beta_nought.shape[1]),order=1)
    noise_eq_beta_nought = np.transpose(noise_eq_beta_nought)
    np.multiply(noise_eq_beta_nought,np.sin(thetainc),out=noise_eq_beta_nought)


    log.info('Computing noise decorrelation step 2 of 3 ...')

    #get amplitud od the slc, include cal factor!
    amp_slc = ndimage.uniform_filter(np.abs(slc*np.sqrt(parameters['calFactor'])),(ml_az,ml_rg))
    # get radar cross section
    # RCS = np.multiply(np.square(amp_slc),np.sin(thetainc-rg_slope))
    # amp_slc = None
    # power = RCS - noise_eq_beta_nought
    # RCS = None
    # SNR = power / noise_eq_beta_nought
    # power = None
    # noise_eq_beta_nought = None
    np.multiply(np.square(amp_slc),np.sin(thetainc-rg_slope),out=amp_slc)
    amp_slc = amp_slc - noise_eq_beta_nought
    amp_slc = amp_slc / noise_eq_beta_nought

    #amp_slc = numexpr.evaluate("(amp_slc**2)*sin(thetainc-rg_slope)")
    #amp_slc = numexpr.evaluate("amp_slc - noise_eq_beta_nought")
    #amp_slc = numexpr.evaluate("amp_slc / noise_eq_beta_nought")


    log.info('Computing noise decorrelation step 3 of 3 ...')

    #calculate SNR decorrelation
    # gama_SNR = SNR/(SNR+1)
    # SNR = None

    amp_slc = amp_slc/(amp_slc+1)
    #amp_slc = numexpr.evaluate("amp_slc/(amp_slc+1)")

    # filter very low gana_SNR
    #gama_SNR[gama_SNR<0] = 0.01
    amp_slc[amp_slc < 0] = 0.01

    #return gama_SNR
    return amp_slc


def compute_corrected_kz_and_slope_dem(parameters,thetainc,dem_slc_dims,bperp):
    """Compute the kz corrected by the dem, the slope of the dem and range decorrelation

    Parameters
    ----------
    parameters : dict
        Dictionary with parameters of the master image.
        Output of function get_params_from_xml()
    thetainc : 2D numpy array
        Inicidence angle
        Ouputput from get_baselines()
    dem_slc_dims : 2D numpy array
        dem in SLC dimensions
    bperp : 2D numpy array
        Perpendicular baseline
        Ouput from get_baselines()

    Returns
    -------
    kz_dem : 2D numpy array
        Vertical wavenumber corrected by the dem
    deco_rg : 2D numpy array
        Range decorrelation
    rg_slope : 2D numpy array
        Slope of the dem ni the range direction

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """



    log = logging.getLogger('compute_coorected_kz_and_slope_dem')
    log.info('Computing corrrected kz and slope of the DEM ...')


    # get solpe of the dem
    # xd1, xd2 = np.gradient(dem_slc_dims)
    # rg_slope = np.arctan2(xd2, parameters['spacing_rg'] / np.sin(thetainc))
    # az_slope = np.arctan2(xd1,parameters['spacing_az'])
    rg_slope = np.gradient(dem_slc_dims,axis=1)

    np.arctan2(rg_slope, parameters['spacing_rg'] / np.sin(thetainc),out=rg_slope)


    ### get range decorrelation and corrected kz
    matrix_range_vectors = np.transpose(np.reshape(np.repeat(parameters['range_vec'], int(parameters['n_az'])),(int(parameters['n_rg']), int(parameters['n_az']))))
    # kz_1d = (4*np.pi/(constants.c/parameters['f0'])) * (bperp/(matrix_range_vectors* np.sin(thetainc)))
    # B_crit = (constants.c / parameters['f0']) * (parameters['cdw'] / constants.c) * matrix_range_vectors * np.tan(thetainc - rg_slope)
    # deco_rg = 1 - np.abs(bperp / B_crit)
    deco_rg = 1 - np.abs(bperp / ((constants.c / parameters['f0']) * (parameters['cdw'] / constants.c) * matrix_range_vectors * np.tan(thetainc - rg_slope)))
    kz_dem = (4 * np.pi / (constants.c / parameters['f0'])) * (bperp / (matrix_range_vectors * np.sin(thetainc - rg_slope)))

    kz_dem = np.abs(kz_dem)


    return kz_dem,deco_rg,rg_slope




def interpolate_forest_height_no_valids_kz(forest_height_radar_coord_kz_invalids,mask_kz_points):
    """Interpolate invalid points due to wrong values of kz

    Parameters
    ----------
    forest_height_radar_coord_kz_invalids : 2D numpy array
        Forest heights
        output from forest_height_inversion()
    mask_kz_points : 2D numpy array
        mask to indicate which pixels (value not equal 0) have been not included in the kz
        Output from processing_tdx_until_coherence()

    Returns
    -------
    forest_height_radar_coord : 2D numpy array
        Forest height with no invalid pixels

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('interpolate_forest_height_no_valids_kz')
    log.info('Interpolate forest in the non-valids kz points ...')

    try:

        if np.sum(mask_kz_points) > 0:

            # grid output image
            grid_x_out, grid_y_out = np.mgrid[0:mask_kz_points.shape[0]:mask_kz_points.shape[0] * 1j,
                                     0:mask_kz_points.shape[1]:mask_kz_points.shape[1] * 1j]
            # get valules and points for grid data
            valid_pos = mask_kz_points == 0
            values = forest_height_radar_coord_kz_invalids[valid_pos]
            points = np.zeros((len(values), 2))
            points[:, 0] = grid_x_out[valid_pos]
            points[:, 1] = grid_y_out[valid_pos]
            forest_height_radar_coord = interpolate.griddata(points, values, (grid_x_out, grid_y_out), method='linear')

        else:
            #if mask points is all 0, it means that we do not have invalid points
            forest_height_radar_coord = forest_height_radar_coord_kz_invalids

    except:
        log.error('Too many invalid point to interpolate')

    return forest_height_radar_coord


def rebin_arbitrary_dims(arr, new_shape,method='mean'):
    """ function to make the rebind of an array using a given method

    If the new size in not a multiple number then we increase the size until it is multiple

    Advice: If it is not strictly necessary, i would use ndimage.zoom (faster and easier)

    Parameters
    ----------
    arr :  2D array
    new_shape : list
        contain the new shape in a list
    method : str
        Method to be used to go from the the size of the original array to the new shape
        'mean','median','max','min'

    Returns
    -------
    new_arr: array with the new dimenions


    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : March 2021


    """


    #==== OPTION 1 using array split to have exactlly the same dimensions==============
    #reshape array with array split, this allows to split into non-multiple size
    arr_reshape = [np.array_split(i.transpose(), new_shape[1]) for i in np.array_split(arr, new_shape[0])]
    #generate new array with the desired dimension

    new_arr = np.zeros((new_shape[0], new_shape[1]),dtype=arr.dtype)
    for num_row, i in enumerate(arr_reshape):
        for num_col, j in enumerate(i):

            if method == 'mean':
                new_arr[num_row, num_col] = np.nanmean(j)
            if method == 'median':
                new_arr[num_row, num_col] = np.nanmedian(j)
            if method == 'max':
                new_arr[num_row, num_col] = np.nanmax(j)
            if method == 'min':
                new_arr[num_row, num_col] = np.nanmin(j)
    #==============================================================================================

   #  #====== OPTION 2 make reshape of the multiple part and then  add the non multiple part
   #  nrows, ncols = arr.shape
   #  nrows_new = new_shape[0]
   #  ncols_new = new_shape[1]
   #
   #  ## Get the rest of pixels in each dimension
   #  end_row = nrows % nrows_new
   #  end_col = ncols % ncols_new
   #  # if it is 0 it means that the dimension is multiple and we take until the last pixel, if not we take until the last pixels to be multiple
   #  if end_row == 0:
   #      end_row = nrows
   #  else:
   #      end_row = end_row * -1
   #  if end_col == 0:
   #      end_col = ncols
   #  else:
   #      end_col = end_col * -1
   #
   #  ## Divide the input array in one with multiple dimensions and the rest
   #  arr1 = arr[0:end_row, 0:end_col]
   #  arr2_rows = arr[end_row:nrows, ::]
   #  arr2_cols = arr[::, end_col:ncols]
   #
   # #Rebin 2D array arr to shape new_shape by averaging."""
   #  shape = (new_shape[0], arr1.shape[0] // new_shape[0], new_shape[1], arr1.shape[1] // new_shape[1])
   #
   #  new_arr = arr1.reshape(shape)
   #  if method == 'mean':
   #      new_arr = np.nanmean(np.nanmean(new_arr, -1), 1)
   #
   #  ##now we include the last pixels correspondiong to the non multiple rows or columns
   #  if len(arr2_rows) > 0:
   #
   #      if method == 'mean':
   #          arr2_rows = np.nanmean(arr2_rows, 0)
   #          # make the mean of the array using split array
   #          new_arr_rows = [np.nanmean(i) for i in np.array_split(arr2_rows, new_shape[1])]
   #
   #      # get the last row of the array concatenated by the previous onw
   #      last_row = np.vstack((new_arr[-1, :], new_arr_rows))
   #
   #      if method == 'mean':
   #          # make the mean
   #          last_row = np.nanmean(last_row, 0)
   #
   #      # change this new row (that includes the rows of the rest with the previous one
   #      new_arr[-1, :] = last_row
   #
   #  ##now we include the last pixels correspondiong to the non multiple cols or columns
   #  if len(arr2_cols) > 0:
   #      if method == 'mean':
   #          arr2_cols = np.nanmean(arr2_cols, 1)
   #          # make the mean of the array with the new shape
   #          new_arr_cols = [np.nanmean(i) for i in np.array_split(arr2_cols, new_shape[0])]
   #
   #      # get the last row of the array concatenated by the previous onw
   #      last_col = np.vstack((new_arr[:, -1], new_arr_cols))
   #
   #      if method == 'mean':
   #          # make the mean
   #          last_col = np.nanmean(last_col, 0)
   #
   #      # change this new col (that includes the cols of the rest with the previous one
   #      new_arr[:, -1] = last_col
   #
   #  ##==============================================================================================

    # #========OPTION 3 to increase the new size until you have a multiple==============
    # nrows,ncols = arr.shape
    # nrows_new = new_shape[0]
    # ncols_new = new_shape[1]
    #
    # # if the dimensions with the new shape are not multiple we take the first multiple value for the new shape
    # while(nrows % nrows_new != 0):
    #     nrows_new = nrows_new + 1
    #
    # while(ncols % ncols_new != 0):
    #     ncols_new = ncols_new + 1
    #
    # #update to the new shape that it is multiple
    # new_shape[0] = nrows_new
    # new_shape[1] = ncols_new
    #
    #
    # """Rebin 2D array arr to shape new_shape by averaging."""
    # shape = (new_shape[0], arr.shape[0] // new_shape[0],
    #          new_shape[1], arr.shape[1] // new_shape[1])
    #
    #
    # new_arr = arr.reshape(shape)
    # if method == 'mean':
    #     new_arr = np.nanmean(np.nanmean(new_arr,-1),1)
    # if method == 'median':
    #     new_arr = np.nanmedian(np.nanmedian(new_arr,-1),1)
    # if method == 'max':
    #         new_arr = np.nanmax(np.nanmax(new_arr, -1), 1)
    # if method == 'min':
    #     new_arr = np.nanmin(np.nanmin(new_arr, -1), 1)
    # #==============================================================================================

    return new_arr




def processing_tdx_until_coherence(inputs,path_image_acquisition):
    """Processing of TDX image from cos until coherecen

    Parameters
    ----------
    inputs: module
        Module from the inputs file used in the GEDI/TDX procesinng
        Before calling the function make import inputs
    path_image_acquisition : str
        Complete path of the folder that contains the TDX image to process

    Returns
    -------
    parameters : dict
        Information related to the master image
    coh_cor : 2D numpy array
        Coherence corrected by the dem
    kz_cor : 2D numpy array
        Kz corrected by the dem
    dem : 3D numpy array
        DEM in the form of a 3D array, where the last dimension represents:
            - (rows, cols,0): Longitude
            - (rows, cols,1): Latitude
            - (rows, cols,2): Height
        output from get_dem()
    dem_limits : dict
        Limits of the DEM.
        It contains the following keys:  {'minlon': ,'maxlon': ,'minlat': ,'maxlat': 0.}
        output from get_dem()
    deco_rg : 2D numpy array
        Range decorrelation
    phase : 2D numpy array
        Interferogram phase
    uw_phase : 2D numpy array
        Interferogram unwrapped phase
    master_image : 2D numpy array
        slc master image
        If the inputs.save_master_slave is False then master_image=None
    slave_image : 2D numpy array
        slc slave image
        If the inputs.save_master_slave is False then slave_image=None
    kz : 2D numpy array
        Original kz (not corrected by the dem)
        this is not the kz used for the forest heigth!
    coherence : 2D numpy array
        Absolute value of the coherence (not corrected by the dem)
        this is not the kz used for the forest heigth!
    dem_slc_dims : 2D numpy array
        DEM used for the processing of the interferogram in the SLC dimensions
    im_filtered_std : 2D numpy array
        Standard deviation of the third highest pixel in a the window omf inputs.resolution


    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('processing_tdx_until_coherece')
    log.info('Pre- Processing of tdx/tsx images until coherence ...')


    # =============== READING TDX PARAMETERS FROM XML =============== #
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    parameters, parameters_slave = get_params_from_xml(path_image_acquisition)
    ####################################################################

    # =============== READING EXTERNAL DEM =============== #
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    dem, dem_xyz, dem_posting, dem_limits, dang = get_dem(inputs.path_dem,inputs.type_dem,parameters, margin_degrees=0.5,NumThreads=inputs.num_threads)
    ####################################################################

    # =============== INTERPOLATION OF ORBITS =============== #
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    parameters['orbit_slc'] = interpol_orbits(parameters, margin=0)
    parameters['orbit'] = interpol_orbits(parameters)
    parameters_slave['orbit_slc'] = interpol_orbits(parameters_slave, margin=0)
    parameters_slave['orbit'] = interpol_orbits(parameters_slave)
    parameters['orbit_active'] = interpol_orbits(parameters,reqTime=parameters_slave['orbit'][:,0],same_coefficients=False,parameters_slave=parameters_slave)
    ####################################################################

    # =============== PARAMETERS FOR BACK-GEOCODED DEM =============== #
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    ## master
    #### get angles
    offnadir, lookangle = get_offnadir_lookangle(parameters, dem,NumThreads=inputs.num_threads)
    # get more parameters back geocoding dem
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    deltat_dem, rd_dem, rs_dem, t0_dem, nrg_dem, naz_dem = get_params_back_geocoding_dem(parameters,dem_posting, offnadir)
    ### slave
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30), 2)) + ' Gb')
    offnadir_slave, lookangle_slave = get_offnadir_lookangle(parameters_slave, dem,NumThreads=inputs.num_threads)
    # get more parameters back geocoding dem
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    deltat_dem_slave, rd_dem_slave, rs_dem_slave, t0_dem_slave, nrg_dem_slave, naz_dem_slave = get_params_back_geocoding_dem(parameters_slave, dem_posting, offnadir_slave)
    # adding same offset to all images
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    rd_dem, rd_dem_slave, t0_dem, t0_dem_slave = add_same_offset_images(parameters, parameters_slave, rd_dem,rd_dem_slave, t0_dem)
    ####################################################################

    # =============== CHANGE DEM TO SLANT-RANGE COORDINATES =============== #
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    rgm, azm = xyz2rgaz(parameters, dem_xyz, dem_limits, deltat_dem, rd_dem, rs_dem, t0_dem, nrg_dem, naz_dem,NumThreads=inputs.num_threads)
    dem_radar_coord = get_dem_height_from_rg_az(rgm, azm, parameters, dem, deltat_dem, rd_dem, rs_dem, t0_dem,nrg_dem, naz_dem)
    # Iterpolation DEM to get the same size as the SLC
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    dem_slc_dims = from_dem_dims_to_slc_dims(parameters, dem_radar_coord, nrg_dem, naz_dem, rd_dem, rs_dem,deltat_dem, t0_dem)
    dem_slc_dims = dem_slc_dims.astype('float32')
    ####################################################################

    # =============== COMPUTATION OF PHASE FLAT IN SLANT-RANGE GEONMETRY =============== #
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    ### compute dem flat earth  -----
    dem_xyz_flat_master = get_dem_xyz_flat_earth(dem, dem_limits, dang,NumThreads=inputs.num_threads)
    ## get master rg az matrices for flat earth
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    rgm_flat_master, azm_flat_master = xyz2rgaz(parameters, dem_xyz_flat_master, dem_limits, deltat_dem, rd_dem,rs_dem, t0_dem, nrg_dem, naz_dem,NumThreads=inputs.num_threads)
    ## get slave rg az matrix for flat earth
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    rgm_flat_slave, azm_flat_slave = xyz2rgaz(parameters_slave, dem_xyz_flat_master, dem_limits,
                                                      deltat_dem_slave, rd_dem_slave, rs_dem_slave, t0_dem_slave,
                                                      nrg_dem_slave, naz_dem_slave, False, True, rs_dem,
                                                      parameters['orbit_active'],NumThreads=inputs.num_threads)
    ## from here following code of get_offsets.pro from line 482 of TAXI with keyword /flatearth
    # convert master matrices to dem positions
    log.debug('Memory usage  ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    slantphaseflat = compute_slant_phase_flat(parameters, parameters_slave, rgm_flat_master, azm_flat_master,
                                                      rgm_flat_slave, nrg_dem, naz_dem, rd_dem, rs_dem, deltat_dem,t0_dem)
    # interpolation to the same size as the SLC
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    slantphaseflat = from_dem_dims_to_slc_dims(parameters, slantphaseflat, nrg_dem, naz_dem, rd_dem, rs_dem,deltat_dem, t0_dem)
    slantphaseflat = slantphaseflat.astype('float32')
    ####################################################################



    # =============== COMPUTATION OF BASELINES =============== #
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    # 1 - get the cartesian coordinates of the range azimuth matrix (i.e dem in cartersians in slant-range geometry)
    dem_xyz_slr = rgz2xyz(parameters, rgm, azm, dem_xyz, deltat_dem, rd_dem, rs_dem, t0_dem, nrg_dem, naz_dem)
    # 2 - get az offset
    # get range azimuth matrices for the slave
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    rgs, azs = xyz2rgaz(parameters_slave, dem_xyz, dem_limits, deltat_dem_slave, rd_dem_slave, rs_dem_slave,
                                t0_dem_slave, nrg_dem_slave, naz_dem_slave, False, True, rs_dem, parameters['orbit'],NumThreads=inputs.num_threads)
    # get offsets in azimuth
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    az_offset,rg_offset,synth_phase = get_offsets(rgm, azm, rgs, azs, parameters,parameters_slave, dem, deltat_dem, rd_dem, rs_dem, t0_dem, nrg_dem,naz_dem)
    # 3. Get parameters related to baselines (bperp, kz, thetainc)  (get_baselines.pro -> get_baselines_dub.pro)
    baseline, bpar, bperp, kz, thetainc = get_baselines(parameters, parameters_slave, dem_xyz_slr, az_offset,deltat_dem, t0_dem)

    # interpolate to the same size as slc
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    bperp = from_dem_dims_to_slc_dims(parameters, bperp, nrg_dem, naz_dem, rd_dem, rs_dem, deltat_dem, t0_dem)
    bperp = bperp.astype('float32')
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30), 2)) + ' Gb')
    kz = from_dem_dims_to_slc_dims(parameters, kz, nrg_dem, naz_dem, rd_dem, rs_dem, deltat_dem, t0_dem)
    kz = kz.astype('float32')
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    thetainc = from_dem_dims_to_slc_dims(parameters, thetainc, nrg_dem, naz_dem, rd_dem, rs_dem, deltat_dem,t0_dem)
    thetainc = thetainc.astype('float32')
    ####################################################################



    # =============== READING TDX/TSX COS FILES =============== #
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    master_image = read_master_slave(path_image_acquisition,parameters,'master')
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    slave_image = read_master_slave(path_image_acquisition, parameters,'slave')
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    # apply calFactor
    #master_image = master_image * np.sqrt(parameters['calFactor'])
    #slave_image = slave_image * np.sqrt(parameters_slave['calFactor'])

    ## save outputs for LEA
    # np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/master_slc_radar_coord.npy', master_image)
    # np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/slave_slc_radar_coord.npy', slave_image)
    # np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/flat_earth_slc_radar_coord.npy', slantphaseflat)
    # np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/kz_radar_coord.npy', kz)


    # =============== COMPUTATION OF INTERFEROGRAM  =============== #
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    #meanheight = np.median(dem[:, :, 2])
    interferogram, coherence = compute_interferogram(inputs.resolution, parameters, master_image, slave_image,slantphaseflat, kz, dem_slc_dims,None,inputs.resolution_slcs)
    #remove residual phase in case we want to save it
    if inputs.save_phase:
        interferogram = remove_residual_flat_earth(interferogram, coherence)


    ## ------- PHASE UNWRAPPING
    if inputs.make_phase_unwrapping:
        try:
            # remove nans from coherence
            coherence[np.isnan(coherence)] = 0
            # WARNING: Based on choi IDL code, we have to test it!
            #Reduce the image to make the phase unwrapping to reduce the computational cost.
            # We take the applyed multilloks
            ml_rg = np.int(np.round(inputs.resolution/((parameters['groundNear']+parameters['groundFar'])/2.)))
            ml_az = np.int(np.round(inputs.resolution/parameters['projectedSpacingAzimuth']))
            # we reduce the orginal multilooks to have a bit of margin respect to the original size
            ml_rg = np.float(ml_rg-2)
            ml_az = np.float(ml_az-2)
            phase = ndimage.zoom(np.angle(coherence), (1/ml_az, 1/ml_rg),order=1)
            abs_coh = ndimage.zoom(np.abs(coherence), (1/ml_az, 1/ml_rg),order=1)
            ## phase unwrapping
            uw_phase = phase_unwrapping(abs_coh, phase, inputs.path_snaphu, inputs.output_path+path_image_acquisition.split('/')[-2]+'/')
            # get again SLC dimensions
            uw_phase = ndimage.zoom(uw_phase, (np.float(coherence.shape[0]/np.float(uw_phase.shape[0])), np.float(coherence.shape[1]/np.float(uw_phase.shape[1]))),order=1)
            ## ------- BASELINE CORRETION
            ## compute plane for the baseline correction
            plane = baseline_correction_using_plane(np.abs(coherence), uw_phase, kz)
            ## remove this plane to the interferogram
            np.multiply(interferogram,np.exp(-plane*kz*1j), out = interferogram)
            # remove this plant to the unwrtapped pahse
            uw_phase  = uw_phase - plane*kz
            # get the phase of the interferogram
            phase = np.angle(interferogram)
        except:
            log.error('Unexpected error in the phase unwrapping')
            log.exception('Error:')
            log.info('Generate a fake unwrapped phase to continue the processing')
            uw_phase = np.zeros(interferogram.shape)
    else:
        # get the phase of the interferogram
        phase = np.angle(interferogram)
        uw_phase = None
    ####################################################################


    # =========================================================================
    # Changhyun approach based on the standard deviation of third highest pixel
    if inputs.make_map_std_for_bias_height:
        im_filtered_std = map_std_for_bias_height(master_image, slave_image, slantphaseflat,kz,rg_slope,parameters,inputs,path_image_acquisition)
    else:
        im_filtered_std = None
    # ==================================================

    #save momory
    if inputs.save_kz == False:
        kz = None
    interferogram = None

    # =============== COMPUTATION OF VOLUMETRIC COHERENCE CORRECTION  =============== #
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    ## 1 - Calculation of DEM corrected kz
    kz_cor, deco_rg, rg_slope = compute_corrected_kz_and_slope_dem(parameters, thetainc,dem_slc_dims, bperp)

    kz_cor = np.clip(kz_cor,inputs.hard_lower_limit_kz,inputs.hard_upper_limit_kz)
    bperp = None
    #dem_slc_dims = None


    # 2- compute noise decorrelation
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    SNR_master = compute_noise_decorrelation(inputs.resolution, parameters, np.copy(master_image), thetainc,rg_slope)
    log.debug('Memory usage ' + str(np.round((psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30) , 2)) + ' Gb')
    SNR_slave = compute_noise_decorrelation(inputs.resolution, parameters_slave, np.copy(slave_image), thetainc,rg_slope)
    rg_slope  = None
    thetainc  = None

    # 3-# correct coherence
    coh_cor = np.clip(np.abs(coherence) / deco_rg / np.sqrt(SNR_master * SNR_slave), 0, 1)

    ## save outputs for LEA
    # np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/SNR_master_coord.npy', SNR_master)
    # np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/SNR_slave_coord.npy', SNR_slave)

    SNR_master  = None
    SNR_slave = None
    if inputs.save_coh:
        coherence  = np.abs(coherence)
    else:
        coherence = None
    ####################################################################


    #check if we want to save the master/slave images
    if inputs.save_master_slave == False:
        master_image = None
        slave_image = None

    #Added kz and coherence mean to the parameters file
    parameters['kz_cor_mean'] = np.nanmean(kz_cor)
    parameters['coh_cor_mean'] = np.nanmean(coh_cor)

    ## save outputs for LEA
    # np.save(inputs.output_path+path_image_acquisition.split('/')[-2]+'/dem_latlon_coord.npy',dem)
    # np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/dem_radar_coord.npy', dem_slc_dims)
    # np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/range_decorrelation_radar_coord.npy', deco_rg)
    # np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/kz_corrected_by_dem_radar_coord.npy', kz_cor)

    #### ADITIONAL DECORRELATION FACTORS ##################
    # Increase of coherence to reduce bias in lower heights
    coh_cor = np.clip(coh_cor/inputs.decorrelation_coherence_before_inversion,0,1)
    #quantization error correction (coherence error of quantization is normally around 3 %)
    coh_cor = coh_cor/inputs.quantization_error
    #remove values that are lower than 0 or heihger than one
    #coh_cor[(coh_cor < 0) | (coh_cor > 1)] = np.nan
    coh_cor = np.clip(coh_cor,0,1)
    #####################################################################################



    return parameters,coh_cor,kz_cor,dem,dem_limits,deco_rg,phase,uw_phase,master_image,slave_image,kz,coherence,dem_slc_dims,im_filtered_std



def map_std_for_bias_height(master_image, slave_image, slantphaseflat, kz, rg_slope, parameters, inputs, path_image_acquisition):
    """Code suggested from changhyun to remove the bias on the height based on the standadrd deviation of the heights


    Parameters
    ----------
    master_image : 2D numpy array
        master image in SLC coordinates
    slave_image : 2D numpy array
        Slave image in SLC coordinates
    slantphaseflat : : 2D numpy array
        Output from compute_slant_phase_flat() with the SLC dimensions
    kz : 2D numpy array
        Vertical wavenumber in SLC coordinates
    dem : 2D numpy array
        DEM in SLC coordinates
    parameters : dict
        Information related to the master image
    inputs: module
        Module from the inputs file used in the GEDI/TDX procesinng
        Before calling the function make import inputs

    Returns
    -------
    im_filtered_std : 2D numpy array
        Standard deviation of the third highest pixel in a the window omf inputs.resolution

    Notes
    -------
    Author : victor Cazcarra-Bes / Changhyun Choi
    Date : June 2021


    """
    log = logging.getLogger('map_std_for_bias_height')
    log.info('Generation a map to deal with the bias in the height...')


    dem = 0.
    # Step 0: compute the interferomgram
    interferogram, coherence = compute_interferogram(inputs.map_std_resolution_coherence, parameters, master_image, slave_image, slantphaseflat, kz, dem, None, 1)
    master_image = None
    slave_image = None
    slantphaseflat = None
    dem = None
    # smooth images
    ml_rg_tdx = np.int(np.round(inputs.low_resolution_filter_height_tdx / ((parameters['groundNear'] + parameters['groundFar']) / 2.)))
    ml_az_tdx = np.int(np.round(inputs.low_resolution_filter_height_tdx / parameters['projectedSpacingAzimuth']))
    interferogram2 = ndimage.uniform_filter(interferogram.real, (ml_az_tdx, ml_rg_tdx)) + (-1) * 1j * ndimage.uniform_filter(interferogram.imag,(ml_az_tdx, ml_rg_tdx))

    ## Step 1 get the phase of the interferogram and correct using plane aprroahch
    log.info('Get the phase of the interferogram ...')
    # plane correction and topography compensation
    interferogram = np.multiply(interferogram, interferogram2)
    interferogram2 = None
    interferogram[abs(coherence) < 0.4] = 0
    
    #### save interferogram
    #np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/inf_radar_coord.npy', interferogram)
    #np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/inf2_radar_coord.npy', interferogram)
    #np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/coh_radar_coord.npy', coherence)

    coherence = None
    mask_slope = np.zeros(interferogram.shape) + 1
    mask_slope[abs(rg_slope) > np.pi/9.0] = np.nan
    #np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/rg_slope_radar_coord.npy', rg_slope)
    rg_slope = None
    interferogram = interferogram*mask_slope
    #np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/mask_slope_radar_coord.npy', mask_slope)
    mask_slope = None


    ## Step 2 convert phase to height
    log.info('Convert to height ...')
    height_tdx = np.angle(interferogram) / kz
    #np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/residual_height_radar_coord.npy', height_tdx)
    interferogram = None
    kz = None

    #Step 3 filter previous result at 120 m and make the difference of the non filtered and the filtered at 120 m
    log.info('Apply filter at low resolution ...')
    #ml_rg_tdx = np.int(np.round(inputs.low_resolution_filter_height_tdx / ((parameters['groundNear'] + parameters['groundFar']) / 2.)))
    #ml_az_tdx = np.int(np.round(inputs.low_resolution_filter_height_tdx / parameters['projectedSpacingAzimuth']))
    #height_tdx_dem_filter = ndimage.uniform_filter(height_tdx_dem,(ml_az_tdx,ml_rg_tdx))
    #height_tdx = height_tdx_dem - height_tdx_dem_filter
    #### save height
    #np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/hgt_av120_radar_coord.npy', height_tdx)
    height_tdx[height_tdx > 25] = 0

    try:
        # first we try a fast method, in case we have problems with the memory we will do it using generic_filter in parallel
        # Step 5. Get the height of the  third highest peak at 25 meters window
        log.info('Get the height of the third highest peak ...')
        ml_rg = np.int(np.round(inputs.resolution_filter_height_peak / ((parameters['groundNear'] + parameters['groundFar']) / 2.)))
        ml_az = np.int(np.round(inputs.resolution_filter_height_peak / parameters['projectedSpacingAzimuth']))
        #make the filter to get the third highest pixel with a "jumping window" (i.e not sliding window) to reduce the number of pixels
        jump_row = int(ml_az/2.5)
        jump_col = int(ml_rg/2.5)
        sel_num = -1*int(ml_rg*ml_az/20.0)
        height_peak = lib_filter.filter_get_highest_pixel(height_tdx,sel_num,ml_az,ml_rg,jump_row,jump_col)
        log.info('N of tallest phase: ' + str(sel_num))
        # # Step 6. Make a std filter of the previous result
        log.info('Make the std filter ...')
        #get the size of the window taking into account that we reduce the number pixels before
        ml_rg = np.int(np.round(inputs.resolution_filter_std / ((parameters['groundNear'] + parameters['groundFar']) / 2.))/jump_col)
        ml_az = np.int(np.round(inputs.resolution_filter_std / parameters['projectedSpacingAzimuth'])/jump_row)
        im_filtered_std = lib_filter.filter_std(height_peak, ml_az, ml_rg, 1, 1)
        #np.save(inputs.output_path + path_image_acquisition.split('/')[-2] + '/im_filtered_std_radar_coord.npy', im_filtered_std)
        ## get the original dimensions
        im_filtered_std = ndimage.zoom(im_filtered_std,(np.float(height_tdx.shape[0])/np.float(im_filtered_std.shape[0]),np.float(height_tdx.shape[1])/np.float(im_filtered_std.shape[1])),order=1)

    except:

        log.info('Get the height of the third highest peak unsing generic filter ...')
        ml_rg = np.int(np.round(inputs.resolution_filter_height_peak / ((parameters['groundNear'] + parameters['groundFar']) / 2.)))
        ml_az = np.int(np.round(inputs.resolution_filter_height_peak / parameters['projectedSpacingAzimuth']))
        height_peak = lib_filter.generic_filter_parallel(height_tdx, inputs.num_threads_generic_filter, lib_filter.get_third_highest_pixel, (ml_az,ml_rg))

        log.info('Make the std filter using generic filter ...')
        ml_rg = np.int(np.round(inputs.resolution_filter_std / ((parameters['groundNear'] + parameters['groundFar']) / 2.)))
        ml_az = np.int(np.round(inputs.resolution_filter_std / parameters['projectedSpacingAzimuth']))
        im_filtered_std = lib_filter.generic_filter_parallel(height_peak, inputs.num_threads_generic_filter, np.std, (ml_az, ml_rg))


    log.info('Generation a map to deal with the bias in the height end')

    return im_filtered_std



def remove_residual_flat_earth(interferogram, coherence):
    """ Remove residual flat earth component for the image by fitting the sine and cosine to real and imaginary parts of interferogram

    Parameters
    ----------
    interferogram : complex image
        interferogram with residual flat earth component

    coherence : complex image
        coherence

    Returns
    -------
    interferogram_cor : complex image
        corrected interferogram with removed flat earth


    Notes
    -------
    Author : Roman Guliaev (roman.guliaev@dlr.de)
    Date : March 2021

    """
    log = logging.getLogger('remove_residual_flat_earth')
    log.info('Remove residual flat earth ...')


    try:

        interferogram1 = np.copy(interferogram) / np.abs(interferogram)

        def test_func_cos(x, a, b, c):
            return c * np.cos(b * x + a)

        def test_func_sin(x, a, b, c):
            return c * np.sin(b * x + a)

        x_data = np.linspace(0, 2 * np.pi, interferogram1.shape[1])


        # calculating mean imag and real interferogram along azimuth for each range
        interferogram2_imag = np.nanmean(interferogram1.imag, axis=0)
        interferogram2_real = np.nanmean(interferogram1.real, axis=0)

        # check for nan values
        x_data_imag = x_data[np.isfinite(interferogram2_imag)]
        interferogram2_imag = interferogram2_imag[np.isfinite(interferogram2_imag)]

        x_data_real = x_data[np.isfinite(interferogram2_real)]
        interferogram2_real = interferogram2_real[np.isfinite(interferogram2_real)]

        # find fit
        params_imag, para2 = optimize.curve_fit(test_func_sin, x_data_imag, interferogram2_imag, p0=[1, 1, .8])
        param_imag = params_imag[1]

        params_real, para2 = optimize.curve_fit(test_func_cos, x_data_real, interferogram2_real, p0=[1, 1, .8])
        param_real = params_real[1]

        sign_phase0 = 1
        if (np.abs(params_imag[0] - params_real[0]) > np.pi / 2 and np.abs(params_imag[0] - params_real[0]) < 3 * np.pi / 2): sign_phase0 = -1

        # taking mean of fit parameter
        flat_frequency = np.mean([np.abs(param_imag), np.abs(param_real)])

        # find the sign of complex exponent
        sign_for_rotation = sign_phase0 * np.sign(params_real[1]) * np.sign(params_real[2]) * np.sign(params_imag[1]) * np.sign(params_imag[2])

        # vector along range
        plane = np.exp(- sign_for_rotation * 1j * x_data * flat_frequency)

        # repeat the same vector for each azimuth
        plane2 = np.repeat([plane], interferogram1.shape[0], axis=0)

        # rotate the interferogram (remove the flat earth)
        interferogram3 = interferogram1 * plane2

        # absolute ground phase compensation
        interferogram_ground = np.angle(np.nanmean(interferogram3[np.abs(coherence > .93)]))
        interferogram_cor = interferogram3 * np.exp(-1j * interferogram_ground)


    except:
        log.error('Un-expected error removing residual flat earth')
        return interferogram

    return interferogram_cor

def read_master_slave(path_image_acquisition,parameters,image_to_read):
    """ Read master or slave image

    Parameters
    ----------
    path_image_acquisition : str
        Complete path of the folder that contains the TDX image to process
    parameters : dict
        Inforamtion related to the image
    image_to_read : str
        string to select the images to read 2 options: 'master' or 'slave'

    Returns
    -------
    slc_image : 2D numpy array
        Single Look complex image

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : November 2020

    """


    log = logging.getLogger('read_master_slave')
    log.info(' Reading '+image_to_read)


    if image_to_read == 'slave':
        if parameters['active_sat'] == 'TSX':
            ###TDX
            slc_image = read_cos(path_image_acquisition + parameters['TDX_name'], parameters)
        else:
            ##TSX
            slc_image = read_cos(path_image_acquisition + parameters['TSX_name'], parameters)

    elif image_to_read == 'master':
        if parameters['active_sat'] == 'TSX':
            ##TSX
            slc_image = read_cos(path_image_acquisition + parameters['TSX_name'], parameters)
        else:
            ###TDX
            slc_image = read_cos(path_image_acquisition + parameters['TDX_name'], parameters)

    else:
        log.error(' Wrong parameter image_to_read: '+image_to_read+'. Must be master or slave')


    return slc_image





def processing_until_forest_height(parameters, coh_cor, kz_cor, dem, common_profile, inputs,make_geocoding=True,use_input_size=False):
    """Get the forest height from the coherences and kz

    Parameters
    ----------
    parameters : dict
        Inforamtion realted to the master image
        Output from processing_tdx_until_coherence()
    coh_cor : 2D numpy array
        Coherence corrected by the dem
        Output from processing_tdx_until_coherence()
    kz_cor : 2D numpy array
        Kz corrected by the den
        Output from processing_tdx_until_coherence()
    dem : 3D numpy array
        DEM in the form of a 3D array, where the last dimension representes:
            - (rows, cols,0): Longitude
            - (rows, cols,1): Latitude
            - (rows, cols,2): Height
        output from get_dem() / Output from processing_tdx_until_coherence()
    common_profile : list of 1D numpy array
        Common profiles generated from GEDI data
    inputs: module
        Module from the inputs file used in the GEDI/TDX procesinng
        Before calling the function make import inputs
    make_geocoding : bool
        Flag to make the geocoding of the result.
            - If True is assumed the inputs kz and coherence are in radar coordintes and the results will be given in lat lon
            - If False it returns the result in the same coordiantes as the incputs coh_cor and kz_cor
    use_input_size : bool
        Flag to use the size of the input data to not compute the height for all pixels.
        - If False The dada is reduced by the corresponding desired pixel spacing in output taking into account the original pixel spacing

    Returns
    -------
    forest_height_geo_lonlat : 2D numpy array
        Forest height
    col_axis_lat_coord: 1D numpy array
        Latitude values for the columns of forest_height_geo_lonlat
    row_axis_lon_coord: 1D numpy array
        Longitude values for the rows of forest_height_geo_lonlat
    lut_kz_coh_heights : 2D numpy array
        lut to make the forest height inversion that relates coh/kz <-> height
        First dimension kz as indicate in kz_lut_axes
        Second dimension height with dimensions:
            - height_vector = np.linspace(inputs.min_height_vector, inputs.max_height_vector, num=inputs.n_elements_height_vector)
    kz_lut_axes : 1D numpy array
        values of kz corresponding to the first dimension of lut_kz_coh_heights


    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('processing_until_forest_height')
    log.info('Pre- Processing of tdx/tsx images until forest height  ...')

    # =============== GENERATION OF 2D LUT COHERENCE/KZ <-> HEIGHT =============== #
    height_vector = np.linspace(inputs.min_height_vector, inputs.max_height_vector, num=inputs.n_elements_height_vector)
    lut_kz_coh_heights_list = []
    for num_common_profile,i_common_profile in enumerate(common_profile):
        lut_kz_coh_heights_aux, kz_lut_axes = get_2dlut_kz_coh_height_from_master_profile(i_common_profile, height_vector,kz_min=0,kz_max=1.2 * np.nanmax(kz_cor),n_elements_lut=5000)
        lut_kz_coh_heights_list.append(lut_kz_coh_heights_aux)

    #Check if the profile is adaptative
    if len(common_profile) > 1:
        #interpolate the luts
        lut_kz_coh_heights = lib_profile.interpolate_luts_different_profiles(inputs,lut_kz_coh_heights_list,kz_lut_axes,height_vector)
    else:
        #Tehre is only one lut as there is no adaptative prfdile
        lut_kz_coh_heights = lut_kz_coh_heights_list[0]

    lut_kz_coh_heights_list = None

    # =============== FOREST HEIGHT INVERSION  =============== #
    #Make the forest height inversion
    forest_height_radar_coord = forest_height_inversion(inputs, kz_cor,coh_cor,parameters,lut_kz_coh_heights, kz_lut_axes,use_input_size_kz_coh=use_input_size)
    ##interpolate forest height in the no valid points of kz
    #forest_height_radar_coord = interpolate_forest_height_no_valids_kz(forest_height_radar_coord_kz_invalids,mask_points_processing_height)
   ####################################################################

    # =============== GEOCODING FOREST HEIGHT RADAR TO LAT/LON COORDINATES  =============== #
    if make_geocoding:
        forest_height_geo_lonlat, col_axis_lat_coord, row_axis_lon_coord = geocoding_radar_image(forest_height_radar_coord,parameters, dem,NumThreads=inputs.num_threads,
                                                                                                    margin=0.05,pixels_spacing=inputs.pixel_spacing_out,pixels_border_to_remove=inputs.pixels_border)

        return forest_height_geo_lonlat, col_axis_lat_coord, row_axis_lon_coord, lut_kz_coh_heights, kz_lut_axes

    else:
        col_axis_lat_coord = None
        row_axis_lon_coord = None

        return forest_height_radar_coord, col_axis_lat_coord, row_axis_lon_coord, lut_kz_coh_heights, kz_lut_axes
    #####################################################################




def generate_kml_for_forest_height(image,col_axis_lat_coord,row_axis_lon_coord,inputs,output_path='',img_fname = 'forest_height.png',kml_fname='forest_height.kml',title_name='KML for forest height'):
    """Generation of a kml and png to be used in google earth

    Parameters
    ----------
    image : 2D numpy array
    col_axis_lat_coord : 1D numpy array
        Array with the values of the lat (columns) coordinates with origin on up left corner
    row_axis_lon_coord : 1D numpy array
        Array with the values of the lon (rows) coordinates with origin on up left corner
    inputs : module
        Input file provided in the GEDI/TDX processing.
    output_path: str, optional
    img_fname: str, optional
        Name of the png image. It should be the same as kml_fname
    kml_fname: str, optional
        Name of the kml file. It should be the same as img_fname
    title_name: str, optional
        Name on the


    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2020

    """

    log = logging.getLogger('generate_kml_for_forest_height')
    log.info('Generate a kml and png files  ...')


    #Transform input image in the correct positions
    image = np.flipud(np.transpose(image))

    ## reduce the size of the image, as we only need this for the kml
    image = ndimage.zoom(image,((image.shape[0]/3.0) / image.shape[0],(image.shape[1]/3.0) / image.shape[1]),order=0, mode='nearest')

    #mask the nan points for transparency
    mask = np.ones((image.shape))
    mask[np.isnan(image)] = 0

    #get invalid points
    water_points = image==INVALID_WATER
    settement_points = image == INVALID_SETTLEMENTS


    image[water_points] = np.nan
    image[settement_points] = np.nan


    #transform to a RGBA image (last dimension is the trasnparency)
    height_vector = np.linspace(inputs.min_height_vector, inputs.max_height_vector, num=inputs.n_elements_height_vector)
    image = image / np.nanmax(height_vector)
    image_png = cm.YlGn(image)
    #Add transparency to nan points
    image_png[:, :, 3] = mask


    # mask settelments (red)
    image_png[settement_points, 0] = 1
    image_png[settement_points, 1] = 0
    image_png[settement_points, 2] = 0
    image_png[settement_points, 3] = 1
    # mask watter (blue)
    image_png[water_points, 0] = 0
    image_png[water_points, 1] = 0
    image_png[water_points, 2] = 1
    image_png[water_points, 3] = 1

    #plt.imsave(output_path + img_fname, image_png,vmin=inputs.min_height_vector,vmax=inputs.max_height_vector,cmap=cmap)
    plt.imsave(output_path + img_fname, image_png, vmin=inputs.min_height_vector, vmax=inputs.max_height_vector)

    #names for the kml
    kml_fname = output_path + kml_fname
    title = title_name
    img_name = 'Geocoded image'

    #Coordinates of the image
    coods = "{},{} {},{} {},{} {},{}".format(np.min(row_axis_lon_coord), np.min(col_axis_lat_coord),
                                             np.max(row_axis_lon_coord),np.min(col_axis_lat_coord),
                                             np.max(row_axis_lon_coord), np.max(col_axis_lat_coord),
                                             np.min(row_axis_lon_coord), np.max(col_axis_lat_coord))

    # Generate kml directly
    root = et.Element("kml")
    root.set("xmlns", "http://www.opengis.net/kml/2.2")
    root.set("xmlns:gx", "http://www.google.com/kml/ext/2.2")
    doc = et.SubElement(root, "Document")
    et.SubElement(doc, "name").text = title
    overlay = et.SubElement(doc, "GroundOverlay")
    et.SubElement(overlay, "name").text = img_name
    et.SubElement(overlay, "open").text = "1"
    icon = et.SubElement(overlay, "Icon")
    et.SubElement(icon, "href").text = img_fname
    llq = et.SubElement(overlay, "gx:LatLonQuad")
    et.SubElement(llq, "coordinates").text = coods
    tree = et.ElementTree(root)
    tree.write(kml_fname)



def compute_error_between_luts(common_profile,inputs,output_path):
    """Computes the errors of the LUTs generated by the extreme profiles used to generate the final LUT.

    Explanation:
        common_profile contains a list of profiles used for different heights. For example 70 profiles from height 0 to height 70.
        From this 70 profiles we have 70 different LUTs where the LUT for height 0 and the LUT for height 70 are the more extrem ones.
        In this function we compute the error between these two extrem LUTs as well as the LUT of the box


    Parameters
    ----------
    common_profile : list
        list of vectors, where each of them correspond to one profile used to generate the global LUT
    inputs : module
        Input file provided in the GEDI/TDX processing.
    output_path : str
        path where the results will be saved

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : May 2021

    """


    log = logging.getLogger('compute_error_between_luts')
    log.info('Compute the erro between luts')

    #ignore warnings here
    np.seterr('ignore')

    #Define number of elements for the height vector used in the performance curve, we use more than the input height vector to get a better sampling
    n_elements_height_vector = inputs.n_elements_height_vector
    max_height_vector = inputs.max_height_vector
    min_height_vector = inputs.min_height_vector
    height_vector = np.linspace(min_height_vector, max_height_vector, num=n_elements_height_vector)


    #generat the lut
    lut_kz_coh_heights_list = []
    for num_common_profile,i_common_profile in enumerate(common_profile):
        lut_kz_coh_heights_aux, kz_lut_axes = get_2dlut_kz_coh_height_from_master_profile(i_common_profile, height_vector,kz_min=0,kz_max=0.3,n_elements_lut=200)
        lut_kz_coh_heights_list.append(lut_kz_coh_heights_aux)

    #Check if the profile is adaptative
    if len(common_profile) > 1:
        #interpolate the luts
        lut_kz_coh_heights = lib_profile.interpolate_luts_different_profiles(inputs,lut_kz_coh_heights_list,kz_lut_axes,height_vector)
    else:
        #There is only one lut as there is no adaptative prfdile
        lut_kz_coh_heights = lut_kz_coh_heights_list[0]



    # get lut box profile
    box_profile = np.ones(inputs.n_elements_height_vector)
    lut_kz_coh_heights_box, kz_lut_axes = get_2dlut_kz_coh_height_from_master_profile(box_profile, height_vector, kz_min=0, kz_max=0.3, n_elements_lut=200)


    lut_kz_coh_heights = np.clip(lut_kz_coh_heights,inputs.hard_lower_limit_coh,1)
    lut_kz_coh_heights_low = np.clip(lut_kz_coh_heights_list[0], inputs.hard_lower_limit_coh, 1)
    lut_kz_coh_heights_up = np.clip(lut_kz_coh_heights_list[-1], inputs.hard_lower_limit_coh, 1)
    lut_kz_coh_heights_box = np.clip(lut_kz_coh_heights_box, inputs.hard_lower_limit_coh, 1)


    error_lut_kz_coh_heights_low = np.zeros((lut_kz_coh_heights.shape[1],lut_kz_coh_heights.shape[0]))
    error_lut_kz_coh_heights_up = np.zeros((lut_kz_coh_heights.shape[1],lut_kz_coh_heights.shape[0]))
    error_lut_kz_coh_heights_box = np.zeros((lut_kz_coh_heights.shape[1],lut_kz_coh_heights.shape[0]))
    


    for pos_kz in range(lut_kz_coh_heights.shape[0]):
        my_int = interpolate.interp1d(lut_kz_coh_heights[pos_kz, :], height_vector, fill_value="extrapolate")
        ## height for the extrems of the profile
        height_lower_bound = my_int(lut_kz_coh_heights_low[pos_kz, :])
        height_upper_bound = my_int(lut_kz_coh_heights_up[pos_kz, :])
        # heights for the box profile
        height_box = my_int(lut_kz_coh_heights_box[pos_kz, :])

        #compute errors
        error_lut_kz_coh_heights_low[:,pos_kz] = np.abs(height_vector - height_lower_bound)
        error_lut_kz_coh_heights_up[:, pos_kz] = np.abs(height_vector - height_upper_bound)
        error_lut_kz_coh_heights_box[:, pos_kz] = np.abs(height_vector - height_box)



    #compute error in %
    for i_kz in range(error_lut_kz_coh_heights_low.shape[1]):
        error_lut_kz_coh_heights_low[:, i_kz] = error_lut_kz_coh_heights_low[:, i_kz] / height_vector[::-1] * 100
        error_lut_kz_coh_heights_up[:, i_kz] = error_lut_kz_coh_heights_up[:, i_kz] / height_vector[::-1] * 100
        error_lut_kz_coh_heights_box[:, i_kz] = error_lut_kz_coh_heights_box[:, i_kz] / height_vector[::-1] * 100


    ## limit the errors to 25 %
    error_lut_kz_coh_heights_low = np.clip(error_lut_kz_coh_heights_low,0,25)
    error_lut_kz_coh_heights_up = np.clip(error_lut_kz_coh_heights_up, 0, 25)
    error_lut_kz_coh_heights_box = np.clip(error_lut_kz_coh_heights_box, 0, 25)



    plt.figure()
    plt.imshow(error_lut_kz_coh_heights_low,aspect='auto',extent=(kz_lut_axes[0],kz_lut_axes[-1],height_vector[-1],height_vector[0]),cmap='jet')
    plt.colorbar()
    plt.title('Error for lower bound')
    plt.xlabel('kz [rad/m]')
    plt.ylabel('Height [m]')
    plt.savefig(output_path + 'error_luts_lower_bound_2d_plot.png', dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(error_lut_kz_coh_heights_up, aspect='auto', extent=(kz_lut_axes[0], kz_lut_axes[-1], height_vector[-1], height_vector[0]), cmap='jet')
    plt.colorbar()
    plt.title('Error for upper bound')
    plt.xlabel('kz [rad/m]')
    plt.ylabel('Height [m]')
    plt.savefig(output_path + 'error_luts_upper_bound_2d_plot.png', dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(error_lut_kz_coh_heights_box, aspect='auto', extent=(kz_lut_axes[0], kz_lut_axes[-1], height_vector[-1], height_vector[0]), cmap='jet')
    plt.colorbar()
    plt.title('Error for box')
    plt.xlabel('kz [rad/m]')
    plt.ylabel('Height [m]')
    plt.savefig(output_path + 'error_luts_box_bound_2d_plot.png', dpi=200, bbox_inches='tight')
    plt.close()

    ## make also some 1d plots
    values_kz_plot = [0.05, 0.1, 0.15, 0.2]
    for i_value_kz_plot in values_kz_plot:
        pos_kz = np.argmin(np.abs(i_value_kz_plot - kz_lut_axes))
        my_int = interpolate.interp1d(lut_kz_coh_heights[pos_kz, :], height_vector, fill_value="extrapolate")
        ## height for the extrems of the profile
        height_lower_bound = my_int(lut_kz_coh_heights_low[pos_kz, :])
        height_upper_bound = my_int(lut_kz_coh_heights_up[pos_kz, :])
        # heights for the box profile
        height_box = my_int(lut_kz_coh_heights_box[pos_kz, :])

        plt.figure()
        plt.plot(height_vector, height_lower_bound, label='Lower bound profiles')
        plt.plot(height_vector, height_upper_bound, label='Upper bound profiles')
        plt.plot(height_vector, height_box, label='box  profiles')
        plt.plot(height_vector, height_vector, c='k', linestyle='--')
        plt.title('Errors between LUTs '+'Kz: ' + str(np.round(kz_lut_axes[pos_kz], 2)))
        plt.xlabel('Height [m]')
        plt.ylabel('Height [m]')
        plt.legend()
        plt.savefig(output_path + 'errors_between_luts_'+'Kz_' + str(np.round(kz_lut_axes[pos_kz], 2))+'.png', dpi=200, bbox_inches='tight')
        plt.close()

    return


def get_min_max_valid_heights(inputs,parameters,common_profile,kz_cor,forest_height,output_path,plot_performance_kz_mean=True):
    """Compute minimum and maximum valid heights depending on the kz

        Parameters
        ----------
        inputs : module
            Input file provided in the GEDI/TDX processing.
        parameters : dict
            Information related to the master image
            Output from processing_tdx_until_coherence()
    	common_profile : list of 1D numpy arrays
    		list with all common profiles for the generation of the Lut for forest height inversion
        kz_cor : 2D numpy array
            Kz in lat lon coordinates
        forest_height_geo_lonlat : 2D numpy array
            Forest height in lat lon coordinates
        output_path : str
            path where some outputs will be saved
        plot_performance_kz_mean : bool
            To generate the performance plot assuming the true profile


        Returns
        -------
        max_valid_height : 2D numpy array
            Maximum valid height for each pixel in lat lon coordinates
        min_valid_height : 2D numpy array
            Maximum valid height for each pixel in lat lon coordinates
        bias : 2D numpy array
            bias in height respect to the performance plot


        Notes
        -------
        Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
        Date : Nov 2020

        """

    log = logging.getLogger('get_min_max_valid_heights')
    log.info('Compute the minimum and maximum valid range of heights step 1 of 2 ..')

    #ignore warnings here
    np.seterr('ignore')

    #DEfine the height vector
    n_elements_height_vector = inputs.n_elements_height_vector
    max_height_vector = inputs.max_height_vector
    min_height_vector = inputs.min_height_vector
    height_vector = np.linspace(min_height_vector, max_height_vector, num=n_elements_height_vector)


    #generat the lut
    lut_kz_coh_heights_list = []
    for num_common_profile,i_common_profile in enumerate(common_profile):
        lut_kz_coh_heights_aux, kz_lut_axes = get_2dlut_kz_coh_height_from_master_profile(i_common_profile, height_vector,kz_min=0,kz_max=1.2 * np.nanmax(kz_cor),n_elements_lut=200)
        lut_kz_coh_heights_list.append(lut_kz_coh_heights_aux)

    #Check if the profile is adaptative
    if len(common_profile) > 1:
        #interpolate the luts
        lut_kz_coh_heights = lib_profile.interpolate_luts_different_profiles(inputs,lut_kz_coh_heights_list,kz_lut_axes,height_vector)
    else:
        #There is only one lut as there is no adaptative prfdile
        lut_kz_coh_heights = lut_kz_coh_heights_list[0]



    lut_kz_coh_heights = np.clip(lut_kz_coh_heights,inputs.hard_lower_limit_coh,1)

    #Define number  of  elements for the height vector used in the performance curve, we use more than the input height vector to get a better sampling
    height_vector = ndimage.zoom(height_vector,3)
    lut_kz_coh_heights = ndimage.zoom(lut_kz_coh_heights,(1,3))
    n_elements_height_vector = len(height_vector)


    plt.figure(1)
    plt.figure(2)
    values_kz_plot = [0.05,0.075, 0.1,0.125,0.15,0.175,0.2]
    for i_value_kz_plot in values_kz_plot:
        pos_kz = np.argmin(np.abs(i_value_kz_plot - kz_lut_axes))
        plt.figure(1)
        plt.plot(height_vector, lut_kz_coh_heights[pos_kz, :], label='Kz: ' + str(np.round(kz_lut_axes[pos_kz], 3)))
        plt.figure(2)
        plt.plot(height_vector, np.gradient(lut_kz_coh_heights[pos_kz, :]), label='Kz: ' + str(np.round(kz_lut_axes[pos_kz], 3)))




    plt.figure(1)
    plt.title('Look-up table used for the performance')
    plt.xlabel('Height [m]')
    plt.ylabel('Coherence')
    plt.legend()
    plt.savefig(output_path + 'lut_kz_coh_height_for_performance_plot.png', dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(2)
    if inputs.slope_coh_lut_kz is not None:
        plt.plot([height_vector[0],height_vector[-1]],[inputs.slope_coh_lut_kz,inputs.slope_coh_lut_kz],'r--')
    plt.title('Gradient Look-up table used for the performance')
    plt.xlabel('Height [m]')
    plt.ylabel('Coherence')
    plt.legend()
    plt.savefig(output_path + 'gradient_lut_kz_coh_height_for_performance_plot.png', dpi=200, bbox_inches='tight')
    plt.close()



    # get number of looks
    ml_rg = np.int(np.round(inputs.resolution/((parameters['groundNear']+parameters['groundFar'])/2.)))
    ml_az = np.int(np.round(inputs.resolution/parameters['projectedSpacingAzimuth']))
    n_of_looks = np.float(ml_rg*ml_az)



    # For simplicity we use the same number of elements for coh and heights (it can be differet)
    n_elements_coh = n_elements_height_vector
    coherence_vector = np.linspace(0, 1, n_elements_coh)
    n_elements_kz = lut_kz_coh_heights.shape[0]

    # Now we inverted the heights and we get the limits for all range of kzs. I. e. We generate a LUT for limits of heights
    # Note: We do this to avoid doing the same procedurre for all kzs of the image.
    maximum_height_lut = np.zeros(n_elements_kz)
    minimum_height_lut = np.zeros(n_elements_kz)
    bias_lut = np.zeros((n_elements_kz,n_elements_coh))


    ## get positions to p
    list_pos_kz_to_plot = []
    for i_value_kz_plot in values_kz_plot:
        pos_kz = np.argmin(np.abs(i_value_kz_plot - kz_lut_axes))
        list_pos_kz_to_plot.append(pos_kz)

    for num_kz in range(n_elements_kz):

        inverted_heights = np.zeros(n_elements_coh)

        # invert the heights for this lut for coherence from 0 to 1
        for num_coh, i_coh in enumerate(coherence_vector):
            pos = np.where(i_coh >= lut_kz_coh_heights[num_kz, :])
            if len(pos[0]) > 0:
                inverted_heights[num_coh] = height_vector[pos[0][0]]  # inverted_heights[num_coh] = lib_tdx.one_pixel_forest_height(kz_cor_pixel, kz_lut_axes, lut_kz_coh_heights, i_coh, height_vector)

        # remove not valid inverted heights
        inverted_heights[inverted_heights <= 0] = np.nan
        height_coherence_measured = np.copy(inverted_heights)
        #height_coherence_measured[::-1]


        #Variance introduces by the Cramer-Rao bound
        for num_coh, i_coh in enumerate(coherence_vector):
            sigma = (1 - np.square(1 - i_coh)) / n_of_looks * 2 * n_elements_height_vector
            gauss = signal.windows.gaussian(n_elements_height_vector * 2, sigma)
            gauss = gauss / np.nansum(gauss)
            gauss = gauss[num_coh:num_coh + n_elements_height_vector]
            height_coherence_measured[num_coh] = np.nansum(inverted_heights * gauss) / np.nansum(gauss + inverted_heights * 0)


        # we account for constant decorrelation
        height_with_decor = np.zeros(n_elements_coh)
        for num_coh, i_coh in enumerate(lut_kz_coh_heights[num_kz, :]):
            #  Note: we use 1 -  because the coherencen of the lut goes from 1 to 0
            modified_coherence = 1 - i_coh * inputs.decorrelation_filter_kz
            pos_height = np.argmin(np.abs(coherence_vector - modified_coherence))
            height_with_decor[num_coh] = height_coherence_measured[pos_height]


        # Compute the bias
        bias = np.clip((np.abs(height_with_decor - height_vector) / height_vector), 0, 1)
        bias_lut[num_kz,:] = bias

        # get the position of the minimum height
        # NOTE: We add the 0.01 to avoid errors due to the sampling of the function
        pos_bias_lower_limit = np.where(bias < (inputs.limit_bias_min_height - 0.01))[0]
        if len(pos_bias_lower_limit)>1:
            #we get the position of the minimum for the first time we find th bias in the curve
            pos_bias_lower_limit = pos_bias_lower_limit[0]
            minimum_height_lut[num_kz] = height_vector[pos_bias_lower_limit]

            #To get the position of the maximum first we find the first position of the curve where the bias is lower than the limit of maximum
            # then we continue the curve to find the next position which will be the limit.
            # Note: We do it like that, because if the limit of minimum height is higher than the limit of the maximum height, then
            #           we will not get the correct maximum height following the curve.
            pos_bias_lower_limit_for_up = np.where(bias < (inputs.limit_bias_max_height - 0.01))[0]

            if len(pos_bias_lower_limit_for_up) > 1:

                pos_bias_lower_limit_for_up = pos_bias_lower_limit_for_up[0]
                pos_bias_upper_limit = np.where(bias[pos_bias_lower_limit_for_up::] > (inputs.limit_bias_max_height + 0.01))[0]
                if len(pos_bias_upper_limit) > 0:
                    maximum_height_from_bias = height_vector[pos_bias_upper_limit[0] + pos_bias_lower_limit_for_up]
                else:
                    maximum_height_from_bias = max_height_vector

                # find minimum peak
                peaks = signal.find_peaks(height_with_decor * -1)
                if len(peaks[0]) > 0:
                    height_pos_minimum = height_with_decor[peaks[0][0]]
                    # pos_max_height_from_minimum = np.argmin(np.abs(height_with_decor[pos_bias_lower_limit:peaks[0][0]]-height_pos_minimum)) + pos_bias_lower_limit
                    pos_max_height_from_minimum = np.where(height_with_decor >= height_pos_minimum)[0][0]
                    maximum_height_from_minimum = height_vector[pos_max_height_from_minimum]
                else:
                    maximum_height_from_minimum = max_height_vector
                    height_pos_minimum = 0

                if maximum_height_from_minimum < maximum_height_from_bias:
                    maximum_height_lut[num_kz] = maximum_height_from_minimum
                else:
                    maximum_height_lut[num_kz] = maximum_height_from_bias

                #get the maximum height due to the hard limit of coherence
                for i_height in range(lut_kz_coh_heights.shape[1]):
                    aux_error = lut_kz_coh_heights[num_kz, i_height] - inputs.hard_lower_limit_coh
                    maximum_height_from_hard_lower_limit_coh = height_vector[i_height]
                    if aux_error < 0.01:
                        break

                #get the maximum due to the slope of the derivated of the LUT
                gradient_lut = np.gradient(lut_kz_coh_heights[num_kz, :])


                # compute minimum
                peaks = signal.find_peaks(np.abs(gradient_lut))

                # check that we have a minimum,
                if (len(peaks[0]) > 0):

                    if inputs.slope_coh_lut_kz is None:
                        #if the slope is none then we take the position of the first minimum as the limit
                        maximum_height_from_slope = height_vector[peaks[0][0]]

                    else:

                        # from the minimum get the positions where the gradient is higher than the limit (no valid points)
                        pos_invalids = np.where(gradient_lut[peaks[0][0]:] > inputs.slope_coh_lut_kz)

                        # Check that the value of gradient for the  minimum peak is already heigher than the limit (inputs.slope_coh_lut_kz), if not it means that the limit is directlly the last position.
                        # Example: If the minimum is located at -0.001 and  inputs.slope_coh_lut_kz == -0.002 it means that we are always in the upper part of th elimit and we take the last position as limit
                        # In other words, if the first position of the invalids is 0, it means that the slope value of the peak of  minimum is higher than the limit (inputs.slope_coh_lut_kz)  and we take the last position as maximum
                        if (len(pos_invalids[0]) > 0) and (pos_invalids[0][0] != 0):
                            # we take the first position of the invalids as the limit of height + the position of the minimum
                            maximum_height_from_slope = height_vector[peaks[0][0] + pos_invalids[0][0]]
                        else:
                            maximum_height_from_slope = height_vector[-1]
                else:
                    # if there is no minimum, it means that the slope is going always down, we take the last positions as maximum
                    maximum_height_from_slope = height_vector[-1]


                if maximum_height_lut[num_kz] > maximum_height_from_hard_lower_limit_coh:
                    maximum_height_lut[num_kz] = maximum_height_from_hard_lower_limit_coh

                if maximum_height_lut[num_kz] > maximum_height_from_slope:
                    maximum_height_lut[num_kz] = maximum_height_from_slope

                #make the plot performance for the mean of the inputs kz
                if plot_performance_kz_mean:
                    if num_kz in list_pos_kz_to_plot:
                        lib_plots.plot_performance_one_kz(height_vector,height_with_decor,minimum_height_lut[num_kz],min_height_vector,
                                                            max_height_vector,maximum_height_from_bias,maximum_height_from_minimum,maximum_height_from_hard_lower_limit_coh,maximum_height_from_slope,
                                                            height_pos_minimum,maximum_height_lut[num_kz],kz_lut_axes[num_kz],bias,inputs,output_path)


    log.info('Compute the minimum and maximum valid range of heights step 2 of 2 ..')

    nrows,ncols = kz_cor.shape
    maximum_height = np.zeros((nrows,ncols))
    minimum_height = np.zeros((nrows, ncols))
    bias = np.zeros((nrows, ncols))

    # We make all rows for one column at the same time
    kz_lut_axes_matrix = np.reshape(np.repeat(kz_lut_axes, nrows), (len(kz_lut_axes), nrows))
    height_vector_matrix = np.reshape(np.repeat(height_vector, nrows), (len(height_vector), nrows))
    kz_cor_aux = np.copy(kz_cor)
    kz_cor_aux[np.isnan(kz_cor)] = 0
    forest_height_aux = np.copy(forest_height)

    forest_height_aux[forest_height_aux==INVALID_SETTLEMENTS] = 0
    forest_height_aux[forest_height_aux==INVALID_WATER] = 0

    for i_col in range(ncols):

        #vector of kzs for one column
        kz_col = kz_cor_aux[:,i_col]
        #get the closes position in the axis of kz for all rows in the corresponding colum
        pos_kz_col = np.argmin(np.abs(kz_col-kz_lut_axes_matrix),0)
        #convert the position to minimum height basec on the previous generated LUT that realtes (kz and minumum height
        maximum_height[:,i_col] = maximum_height_lut[pos_kz_col]
        minimum_height[:,i_col] = minimum_height_lut[pos_kz_col]

        #compute the bias for all pixels
        forest_col = forest_height[:, i_col]
        pos_forest_height_col = np.argmin(np.abs(forest_col-height_vector_matrix),0)
        bias[:,i_col] = bias_lut[pos_kz_col,pos_forest_height_col]



    #We limit the height to the the input max height/min height used in the processing
    maximum_height = np.clip(maximum_height,inputs.min_height_vector,inputs.max_height_vector)
    minimum_height = np.clip(minimum_height, inputs.min_height_vector, inputs.max_height_vector)
    #add same nanas as kz_cor
    minimum_height[np.isnan(kz_cor)] = np.nan
    maximum_height[np.isnan(kz_cor)] = np.nan
    bias[np.isnan(kz_cor)] = np.nan

    bias[forest_height==INVALID_WATER] =INVALID_WATER
    bias[forest_height == INVALID_SETTLEMENTS] = INVALID_SETTLEMENTS
    minimum_height[forest_height==INVALID_WATER] =INVALID_WATER
    minimum_height[forest_height == INVALID_SETTLEMENTS] = INVALID_SETTLEMENTS
    maximum_height[forest_height==INVALID_WATER] =INVALID_WATER
    maximum_height[forest_height == INVALID_SETTLEMENTS] = INVALID_SETTLEMENTS

    log.info('Compute the minimum and maximum valid range of heights ok!')

    return maximum_height,minimum_height,bias




def compute_masks(inputs,output_path,common_profile,kz_cor,coh_cor,forest_height_geo_lonlat,parameters):
    """Compute the mask of valid/non-calid pixels for kz and coherence

    Parameters
    ----------
    inputs : module
        Input file provided in the GEDI/TDX processing.
    output_path : str
        path where some outputs will be saved
	common_profile : list of 1D numpy arrays
		list with all common profiles for the generation of the Lut for forest height inversion
    kz_cor : 2D numpy array
        Kz in lat lon coordinates
    coh_cor : 2D numpy array
        Coherence in lat lon coordinates
    forest_height_geo_lonlat : 2D numpy array
        Forest height in lat lon coordinates
    parameters : dict
        Infomation related to the master image
        Output from processing_tdx_until_coherence()

    Returns
    -------
    max_valid_height : 2D numpy array
        Maximum valid height for each pixel in lat lon coordinates
    min_valid_height : 2D numpy array
        Maximum valid height for each pixel in lat lon coordinates
    bias : 2D numpy array
        bias in height respect to the performance plot
    mask_kz : 2D numpy array
        Binary matrix with 0 valid and 1 non-valid
        It contains 1 (non-valid) if the forest height estimated is NOT between the limits min_valid_height and max_valid_height
    mask_coh : 2D numpy array
        Binary matrix with 0 valid and 1 non-valid
        It contains 1 (non-valid) if the coherence is lower than inputs.hard_lower_limit_coh

    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : Nov 2020

    """


    log = logging.getLogger('compute_masks')
    log.info('Compute kz and coherence masks ...')


    #### FILTERIN OF kz ##########################
    # Get maximum and minimum valid height and the bias respect to the ideal performance
    max_valid_height, min_valid_height, bias = get_min_max_valid_heights(inputs, parameters, common_profile, kz_cor, forest_height_geo_lonlat, output_path, plot_performance_kz_mean=True)
    # generate a mask with the values of max and min valid kz
    mask_kz = np.ones(kz_cor.shape)
    valid_pos = (forest_height_geo_lonlat > min_valid_height) * (forest_height_geo_lonlat < max_valid_height)
    mask_kz[valid_pos] = 0
    mask_kz[valid_pos] = 0
    mask_kz[np.isnan(kz_cor)] = np.nan
    ##################

    #### FILTERING OF COHERENCE ##########################
    mask_coh = np.zeros(coh_cor.shape)
    # We remove all coherence below certain height
    mask_coh[coh_cor < inputs.hard_lower_limit_coh] = 1
    mask_coh[np.isnan(coh_cor)] = np.nan
    #####################################################

    return max_valid_height, min_valid_height, bias, mask_kz, mask_coh

    log.info('Compute kz and coherence masks ok!')



def phase_unwrapping(coh,phase,path_snaphu,path_files_snaphu):
    """Phase unwrapping unsing snaphu

    Warnings:
        - Now it uses a basic file configuration, ther eis more info on how to properlly fill the config file in TAXI (make_conf_file.pro)

    Parameters
    ----------
    coh : 2D numpy array
        Absolute value of the coherence
    phase : 2D numpy arary
        Coherence phase
    path_snaphu : str
        paht where snaphu is locate
    path_files_snaphu: str
        path where temporary file will be saved

    Returns
    -------
    uw_phase : 2D numpy array
        unwrapped phase


    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : February 2021


    """


    log = logging.getLogger('phase_unwrapping')
    log.info('Compute the phase unwrapping using snaphu ...')

    # save inputs in the binary format for the snap
    phase_file = open(path_files_snaphu + 'phase.dat', 'wb')
    phase_file.write(np.asarray(phase, 'float32'))
    phase_file.close()

    coh_file = open(path_files_snaphu + 'coh.dat', 'wb')
    coh_file.write(np.asarray(coh, 'float32'))
    coh_file.close()

    line_length=coh.shape[1]
    ## generate configuration txt file for the snap
    txt_file = open(path_files_snaphu + 'sanphu_config.txt', 'w+')
    txt_file.write('#############################################\n')
    txt_file.write('# File input and output and runtime options #\n')
    txt_file.write('#############################################\n')
    txt_file.write('#\n')
    txt_file.write('# Input file name\n')
    txt_file.write('#\n')
    txt_file.write('INFILE ' + path_files_snaphu + 'phase.dat\n')
    txt_file.write('#\n')
    txt_file.write('# Input file line length\n')
    txt_file.write('#\n')
    txt_file.write('LINELENGTH '+str(int(line_length))+'\n')
    txt_file.write('#########################\n')
    txt_file.write('# Unwrapping parameters #\n')
    txt_file.write('#########################\n')
    txt_file.write('STATCOSTMODE TOPO\n')
    txt_file.write('#VERBOSE TRUE\n')
    txt_file.write('###############\n')
    txt_file.write('# Input files #\n')
    txt_file.write('###############\n')
    txt_file.write('CORRFILE ' + path_files_snaphu + 'coh.dat\n')
    txt_file.write('################\n')
    txt_file.write('# Output files #\n')
    txt_file.write('################\n')
    txt_file.write('OUTFILE ' + path_files_snaphu + 'uw_phase.dat\n')
    txt_file.write('LOGFILE ' + path_files_snaphu + 'snaphu.log\n')
    txt_file.write('################\n')
    txt_file.write('# File formats #\n')
    txt_file.write('################\n')
    txt_file.write('INFILEFORMAT FLOAT_DATA\n')
    txt_file.write('CORRFILEFORMAT FLOAT_DATA\n')
    txt_file.write('OUTFILEFORMAT FLOAT_DATA\n')
    txt_file.write('################\n')
    txt_file.close()

    # call snaphu
    os.system(path_snaphu+'snaphu -f ' + path_files_snaphu + 'sanphu_config.txt')

    ##read unwrapped phase computed with snaphu
    with open(path_files_snaphu + 'uw_phase.dat', mode='rb') as file:  # b is important -> binary
        fileContent = file.read()
    uw_phase = struct.unpack("f" * (len(fileContent) // 4), fileContent)
    uw_phase = np.array(uw_phase)
    uw_phase = uw_phase.reshape(phase.shape[0], phase.shape[1])

    ##remove tmp files
    os.remove(path_files_snaphu+'sanphu_config.txt')
    os.remove(path_files_snaphu + 'phase.dat')
    os.remove(path_files_snaphu + 'coh.dat')
    os.remove(path_files_snaphu + 'uw_phase.dat')

    return uw_phase


def baseline_correction_using_plane(coh_ab,uw_phase,kz):
    """ Baseline correction based on a plane

    WARNINGS:
        - From choi idl code
        - We should really check with TAXI the baseline correction for a better processing

    Parameters
    ----------
    coh_ab : 2D numpy array
        absolute value of the cohrece
    uw_phase : 2D numpy array
        unwrapped phase
    kz : 2D numpy array
        vertical wavenumber

    Returns
    -------
    plane : 2D numpy array
        Plane with the correction to be applyed to the interferogram


    Notes
    -------
    Author : Victor Cazcarra-Bes (victor.cazcarrabes@dlr.de)
    Date : February 2021


    """

    log = logging.getLogger('baseline_correction_using_plane')
    log.info('Compute the baseline correction using a plane ...')

    z_res  = uw_phase / kz
    cal_points = np.where(coh_ab > 0.95)
    residual = z_res[cal_points[0], cal_points[1]]
    HH = np.asarray(np.vstack([cal_points[0], cal_points[1], np.ones(len(cal_points[0]))]), 'float64')
    cc1 = np.matmul(HH, np.transpose(HH))
    cc2 = np.linalg.inv(cc1)
    cc3 = np.matmul(cc2, HH)
    coef = np.matmul(cc3, residual)
    rgmesh,azmesh = np.meshgrid(range(coh_ab.shape[1]), range(coh_ab.shape[0]))
    #plane = coef[0] * rgmesh + coef[1] * azmesh + coef[2]
    plane = coef[0] *azmesh  + coef[1] * rgmesh + coef[2]



    return plane



