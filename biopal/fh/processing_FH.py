# SPDX-FileCopyrightText: BioPAL <biopal@esa.int>
# SPDX-License-Identifier: MIT

import numpy as np
import os
import logging
from numpy import linalg as LA
from scipy.signal import medfilt2d
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from importlib import reload

from biopal.statistics.utility_statistics import (
    main_correlation_estimation_SR,
    MPMBshuffle,
    main_correlation_estimation_SSF_SR,
)

###
from biopal.fh.processing_FH_add import (
    do_calc_height_single_bas,
    do_calc_height_dual_bas,
    calc_lut_kh_mu,
    set_param_dict,
    profile_dependent_part,
    error_model_kzh
)
###

def volume_decorrelation_lut(num_baselines, vertical_wavenumber, model_parameters):
    """
    Generation of a look-up table containing, for each value of height, extintion
    and baseline, the possible values of volume decorrelation (the decorrelation caused
    by the different projection of the vertical component of the scatterer reflectivity
    spectrum into the interferometric baselines images) [1][2]

    [1] A. Moreira, P. Prats-Iraola, M. Younis, G. Krieger, I. Hajnsek and K. P. Papathanassiou,"A tutorial on synthetic aperture radar," in IEEE Geoscience and Remote Sensing Magazine, vol. 1, no. 1, pp. 6-43, March 2013, doi: 10.1109/MGRS.2013.2248301.
    [2] K. P. Papathanassiou and S. R. Cloude, "Single-baseline polarimetric SAR interferometry," in IEEE Transactions on Geoscience and Remote Sensing, vol. 39, no. 11, pp. 2352-2363, Nov. 2001, doi: 10.1109/36.964971.

    """

    # LUT
    Nh = model_parameters.maximum_height  # 61
    Ns = model_parameters.number_of_extinction_value  # 51
    heights = np.arange(Nh)  # heights    [m]
    extinctions = np.concatenate(
        (np.tan(np.arange(Ns - 1) / (Ns - 1) * np.pi / 2.0) / 30.0, np.array([np.inf])), axis=0
    )  # extinctions    [m^(-1)]

    LUT = np.zeros((Nh, Ns, num_baselines), dtype=np.complex64)

    p = 2.0 * extinctions.reshape((Ns, 1)) * np.ones((1, num_baselines))
    q = np.ones((Ns, 1)) * (1j * vertical_wavenumber).reshape((1, num_baselines))
    q0 = q[0, :]
    p1 = p + q
    for hi in np.arange(1, Nh):  # do not cycle hi = 0
        LU = np.zeros((Ns, num_baselines), dtype=np.complex64)
        LU[1:-1, :] = (
                p[1:-1, :]
                / p1[1:-1, :]
                * (np.exp(p1[1:-1, :] * heights[hi]) - 1.0)
                / (np.exp(p[1:-1, :] * heights[hi]) - 1.0)
        )
        LU[0, :] = (np.exp(q0 * heights[hi]) - 1.0) / (q0 * heights[hi])  # for zero extinction
        LU[-1, :] = np.exp(q0 * heights[hi])  # for inifinite extinction
        LUT[hi, :, :] = LU

    LUT[0, :, :] = np.ones((Ns, num_baselines))  # zero height

    return LUT, extinctions


def estimate_height_core(PI, kz, offnadir, slope, model_parameters, LUT, param_dict,flags_dict, biases,high_coherence_threshold,num_baselines_key="dual"):
    """This function returns forest height, extinction, ground to volume ratio,
       temporal decorrelations.
    """

    """
    Parameters
    ----------


    PI: complex array 3x3*3: num_pol*num_pol*num_bas
        PI matrix for 3 baselines

    kz: float
        kz baseline

    offnadir: float
        off nadir angle
    slope: float
        surface slope angle
    
    LUT: 2d array
        look up table for kh (height*kz) and mu (G/V ratio)
        
    param_dict: dictionary
        parameters for the inversion
        param_dict = { "height_vec_norm": , "kh_vec": ,  "coef_vec_mu": , "temp_vec": }
    
    flags_dict: idct
    
    biases: float array
        pre calculated biases from error model

    high_coherence_threshold: int
        thresholding the high coherence values and preparing a mask (ambiguous estimation) . 
        the thresholded values are reflected in bias map with nan values
  
    num_baselines_key: str
        "dual" or "single" 
        "dual" is default 
    
    calc_meth: str
        "line", "mu", "coherence" or "pols2"
        "line" is default



    Returns
    -------
    output: list
           output_height,
            output_extinction,
            output_GVratio,
            output_gammaT1,
            output_gammaT2,
            output_gammaT3,
            bias,
    """



    if num_baselines_key == "single":

        out_dict = do_calc_height_single_bas(PI, kz, offnadir, LUT, param_dict,flags_dict,biases,high_coherence_threshold,num_bas=1,calc_meth=flags_dict["calc_meth"],error_model_flag=0)
        height = out_dict["height"]
        mu = out_dict["mu"]
        bias=out_dict["bias"]

        output_height = height
        output_extinction = np.nan
        output_GVratio = mu

        output_gammaT1 = np.nan
        output_gammaT2 = np.nan
        output_gammaT3 = np.nan

        output = [
            output_height,
            output_extinction,
            output_GVratio,
            output_gammaT1,
            output_gammaT2,
            output_gammaT3,
            bias,
        ]

    elif num_baselines_key == "dual":
        out_dict = do_calc_height_dual_bas(PI, kz, offnadir, LUT, param_dict, flags_dict, biases,high_coherence_threshold)
        height = out_dict["height"]
        mu = out_dict["mu"]
        bias = out_dict["bias"]

        output_height = height
        output_extinction = np.nan
        output_GVratio = mu

        output_gammaT1 = out_dict["t1"]
        output_gammaT2 = out_dict["t2"]
        output_gammaT3 = np.nan

        output = [
            output_height,
            output_extinction,
            output_GVratio,
            output_gammaT1,
            output_gammaT2,
            output_gammaT3,
            bias,
        ]
    ##roman end

    else:
        # multi baselines processing

        # The furthest point from the ground position is found.
        N_ch = PI.shape[0]
        num_baselines = PI.shape[2]

        I = np.eye(N_ch)  # Identity matrix
        gammav0 = np.zeros(num_baselines, dtype=np.complex64)
        for j in np.arange(num_baselines):
            P = PI[:, :, j] - I
            B = P @ np.conjugate(np.transpose(P))
            w, U = LA.eig(B)
            max_index = np.argmax(w, axis=0)
            w = U[:, max_index].reshape((1, N_ch))
            gammav0[j] = np.conjugate(w) @ P @ np.transpose(w)
            gammav0[j] = gammav0[j] + 1.0


        if np.sum(np.abs(gammav0) >= 1.0) >= 1:
            return [0, 0, np.inf, 1, 1, 1]

        # Solution Search
        # determine maximum mu
        a = np.abs(gammav0 - 1) ** 2
        b = 2.0 * np.real((gammav0 - 1) * np.conjugate(gammav0))
        c = np.abs(gammav0) ** 2 - 1.0
        mumax = np.min((-b + np.sqrt(b ** 2 - 4.0 * a * c)) / 2.0 / a)
        # Depending on baseline, the maximum mu differs.
        # The search should be performed only to the smallest value of the maximal mu's.

        Nmu = model_parameters.number_of_ground_volume_ratio_value  # 100 #number of mu's to be tested.
        Ngt = model_parameters.number_of_temporal_decorrelation_value  # 21  #number of gamma_T's to be tested. >1

        mini = np.zeros(Nmu)
        minima = np.zeros((Ngt, Ngt, Ngt))
        hgt = np.zeros((Ngt, Ngt, Ngt), dtype=np.int32)
        ext = np.zeros((Ngt, Ngt, Ngt), dtype=np.int32)

        for i in np.arange(Nmu):
            mu = i / Nmu * mumax
            gv = mu * (gammav0 - 1.0) + gammav0  # volume only points under given mu.
            distances = np.zeros((Nh, Ns))
            for j in np.arange(num_baselines):
                distances += np.abs(LUT[:, :, j] - gv[j]) ** 2
                # Sum of the distances to each (h, /sigma) pairs.

            mini[i] = np.min(distances)
        subscript = np.argmin(mini)
        mu = float(subscript) / Nmu * mumax
        gv = mu * (gammav0 - 1.0) + gammav0

        gammaT1 = np.arange(Ngt) / (Ngt - 1) * (1.0 - np.abs(gv[0])) + np.abs(gv[0])
        gammaT2 = np.arange(Ngt) / (Ngt - 1) * (1.0 - np.abs(gv[1])) + np.abs(gv[1])
        gammaT3 = np.arange(Ngt) / (Ngt - 1) * (1.0 - np.abs(gv[2])) + np.abs(gv[2])

        for gt1 in np.arange(Ngt):
            for gt2 in np.arange(Ngt):
                for gt3 in np.arange(Ngt):
                    distances = np.zeros((Nh, Ns))
                    gammav0gt = gv / [gammaT1[gt1], gammaT2[gt2], gammaT3[gt3]]
                    # gamma volumes after compensation of temporal decorreltaions.

                    for j in np.arange(num_baselines):
                        distances += np.abs(LUT[:, :, j] - gammav0gt[j]) ** 2
                    # Sum of the distances to each (h, /sigma) pairs.

                    minima[gt3, gt2, gt1] = np.min(distances)
                    subscribe = np.argmin(distances)

                    # Pair of (h, /sigma) of the minimum distance.
                    hgt[gt3, gt2, gt1] = subscribe / Ns
                    ext[gt3, gt2, gt1] = subscribe % Ns

        # Assigning the results.
        subscript = np.argmin(minima)
        pos = np.unravel_index(subscript, minima.shape)

        output_height = heights[hgt[pos]]
        output_extinction = extinctions[ext[pos]]
        output_GVratio = mu

        output_gammaT1 = gammaT1[pos[2]]
        output_gammaT2 = gammaT2[pos[1]]
        output_gammaT3 = gammaT3[pos[0]]

        output = [
            output_height,
            output_extinction,
            output_GVratio,
            output_gammaT1,
            output_gammaT2,
            output_gammaT3,

        ]

    return output


def estimate_height(
        data_stack,
        cov_est_window_size,
        pixel_spacing_slant_rg,
        pixel_spacing_az,
        incidence_angle_rad,
        carrier_frequency_hz,
        range_bandwidth_hz,
        vertical_wavenumber_stack,
        fh_proc_conf,
        R=None,
        look_angles=None,
        ground_slope=None,
):
    # data_stack is a dictionary of two nested dictionaries composed as:
    # data_stack[ acquisition_name ][ polarization ]

    num_acq = len(data_stack)
    acq_names = list(data_stack.keys())
    first_acq_dict = data_stack[acq_names[0]]
    pol_names = list(first_acq_dict.keys())
    num_pols = len(pol_names)
    Nrg, Naz = first_acq_dict[pol_names[0]].shape

    num_baselines = int((num_acq * (num_acq - 1)) / 2)

    # terrain = np.zeros( (Nrg, Naz) )
    if fh_proc_conf.spectral_shift_filtering:
        if R is None or look_angles is None or ground_slope is None:
            raise RuntimeError(
                'FH: when spectral shift filtering is enabled, "R", "look_angles" and "ground_slope" inputs should be specified.'
            )
        (MPMB_correlation, rg_vec_subs, az_vec_subs, subs_F_r, subs_F_a,) = main_correlation_estimation_SSF_SR(
            data_stack,
            cov_est_window_size,
            pixel_spacing_slant_rg,
            pixel_spacing_az,
            incidence_angle_rad,
            R,
            look_angles,
            ground_slope,
            carrier_frequency_hz,
            range_bandwidth_hz,
        )

    else:
        if R is not None or look_angles is not None or ground_slope is not None:
            logging.warning(
                'FH: when spectral shift filtering is disabled, "R", "look_angles" and "ground_slope" inputs are not used.'
            )
        (MPMB_correlation, rg_vec_subs, az_vec_subs, subs_F_r, subs_F_a,) = main_correlation_estimation_SR(
            data_stack,
            cov_est_window_size,
            pixel_spacing_slant_rg,
            pixel_spacing_az,
            incidence_angle_rad,
            carrier_frequency_hz,
            range_bandwidth_hz,
        )

    MBMP_correlation = MPMBshuffle(MPMB_correlation, rg_vec_subs, az_vec_subs, num_pols, num_acq)

    del MPMB_correlation

    Nrg_subs = rg_vec_subs.size
    Naz_subs = az_vec_subs.size
    heightmap = np.zeros((Nrg_subs, Naz_subs))
    extinctionmap = np.zeros((Nrg_subs, Naz_subs))
    ratiomap = np.zeros((Nrg_subs, Naz_subs))
    gammaT1map = np.zeros((Nrg_subs, Naz_subs))
    gammaT2map = np.zeros((Nrg_subs, Naz_subs))
    gammaT3map = np.zeros((Nrg_subs, Naz_subs))
    biasmap = np.zeros((Nrg_subs, Naz_subs))
    vertical_wavenumber = np.zeros((Nrg_subs, Naz_subs, num_baselines))

    ### fixed profile: should be different for each scene. The one below is for Lope
    profile = np.array([0.10911587, 0.10880648, 0.10822586, 0.1071056, 0.10551814,
                        0.10366636, 0.10134622, 0.0988764, 0.0961876, 0.09340365,
                        0.09059029, 0.08783611, 0.0851811, 0.08271189, 0.08048382,
                        0.07850709, 0.07686072, 0.07554259, 0.07458215, 0.07395816,
                        0.07371466, 0.07377687, 0.07416759, 0.07484928, 0.07578662,
                        0.07699981, 0.07839833, 0.0800179, 0.08181661, 0.08378157,
                        0.08592468, 0.0882111, 0.09069071, 0.09335082, 0.09621105,
                        0.09927385, 0.10255558, 0.10607333, 0.10984397, 0.11387994,
                        0.11812421, 0.12260777, 0.1272714, 0.13208926, 0.13700284,
                        0.14195009, 0.14685566, 0.15163967, 0.15620753, 0.16044443,
                        0.16431142, 0.16761106, 0.17034308, 0.17236208, 0.17364381,
                        0.17404525, 0.17363563, 0.17230766, 0.16996441, 0.16690083,
                        0.16272531, 0.15793168, 0.15226306, 0.14587046, 0.1390702,
                        0.13157091, 0.12386631, 0.11584843, 0.10774125, 0.09958973])
    ###


    #setting inversion parameters in a dictionary
    param_dict=set_param_dict()


    """
    default flags:
    num_baselines_key="dual"   #"single"#    number of baselines for the inversion
    calc_meth="line"           # "mu"  #"coherence"#"pols2"  # inversion method 
    rg_deco_flag = 1           #appying range decorrelation
    gauss_flag=0               #no fitting the tomo profile with a gaussian
    coh_opt_flag=0             #no search for maximum eigenvalue of pi matrix
    min_ground_flag=0          #no search for minimum ground
    """

    flags_dict={"num_baselines_key": "dual" ,"calc_meth": "line" ,"rg_deco_flag": 1 ,"gauss_flag": 0,"coh_opt_flag": 0,"min_ground_flag": 0,"high_coherence_threshold_flag":1}

    #profile dependent function
    profile,biases,high_coherence_threshold=profile_dependent_part(profile,param_dict,flags_dict)

    #calculation of the LUT
    LUT = calc_lut_kh_mu(profile, param_dict)


    Nrg_subs_string = str(Nrg_subs)
    for rg_sub_idx in np.arange(Nrg_subs):

        for az_sub_idx in np.arange(Naz_subs):
            logging.info("   Heigth step " + str(rg_sub_idx + 1) + " of " + str(Nrg_subs) + "   Heigth step " + str(
                az_sub_idx + 1) + " of " + str(Naz_subs))

            # for az_sub_idx in np.arange(Naz_subs):
            current_vertical_wavenumber = np.zeros((num_acq, 1))
            for b_idx, stack_curr in enumerate(vertical_wavenumber_stack.values()):
                current_vertical_wavenumber[b_idx] = stack_curr[rg_vec_subs[rg_sub_idx], az_vec_subs[az_sub_idx]]

            current_vertical_wavenumber = current_vertical_wavenumber - current_vertical_wavenumber.T
            current_correlation = MBMP_correlation[:, :, rg_sub_idx, az_sub_idx]
            current_correlation[np.abs(current_correlation) > 1] = np.exp(
                1j * np.angle(current_correlation[np.abs(current_correlation) > 1])
            )

            PI = np.zeros((num_pols, num_pols, num_baselines), dtype=np.complex64)
            n = 0
            for i in np.arange(num_acq - 1):
                for j in np.arange(i + 1, num_acq):
                    PI[:, :, n] = current_correlation[
                                  i * num_pols: num_pols + i * num_pols, j * num_pols: num_pols + j * num_pols,
                                  ]
                    vertical_wavenumber[rg_sub_idx, az_sub_idx, n] = current_vertical_wavenumber[i, j]
                    n += 1

            if np.any(np.isnan(current_vertical_wavenumber)) + np.any(np.isnan(current_correlation)):
                heightmap[rg_sub_idx, az_sub_idx] = np.NaN
                extinctionmap[rg_sub_idx, az_sub_idx] = np.NaN
                ratiomap[rg_sub_idx, az_sub_idx] = np.NaN
                gammaT1map[rg_sub_idx, az_sub_idx] = np.NaN
                gammaT2map[rg_sub_idx, az_sub_idx] = np.NaN
                gammaT3map[rg_sub_idx, az_sub_idx] = np.NaN
                continue

            # LUT, extinctions = volume_decorrelation_lut(
            #     PI.shape[2], vertical_wavenumber[rg_sub_idx, az_sub_idx, :], fh_proc_conf.model_parameters
            # )

            new_k = list(look_angles.keys())[0]
            look_angles1 = look_angles[new_k]


            output = estimate_height_core(PI, vertical_wavenumber[rg_sub_idx, az_sub_idx, :],
                                          look_angles1[rg_sub_idx, az_sub_idx], ground_slope[rg_sub_idx, az_sub_idx],
                                          fh_proc_conf.model_parameters, LUT, param_dict,flags_dict,biases,high_coherence_threshold,
                                          num_baselines_key=flags_dict["num_baselines_key"])

            heightmap[rg_sub_idx, az_sub_idx] = output[0]
            extinctionmap[rg_sub_idx, az_sub_idx] = output[1]
            ratiomap[rg_sub_idx, az_sub_idx] = output[2]
            gammaT1map[rg_sub_idx, az_sub_idx] = output[3]
            gammaT2map[rg_sub_idx, az_sub_idx] = output[4]
            gammaT3map[rg_sub_idx, az_sub_idx] = output[5]
            biasmap[rg_sub_idx, az_sub_idx] = output[6]

    logging.info("   performing median filtering of height products...")
    heightmap = medfilt2d(heightmap, kernel_size=fh_proc_conf.median_factor)
    extinctionmap = medfilt2d(extinctionmap, kernel_size=fh_proc_conf.median_factor)
    ratiomap = medfilt2d(ratiomap, kernel_size=fh_proc_conf.median_factor)
    gammaT1map = medfilt2d(gammaT1map, kernel_size=fh_proc_conf.median_factor)
    gammaT2map = medfilt2d(gammaT2map, kernel_size=fh_proc_conf.median_factor)
    gammaT3map = medfilt2d(gammaT3map, kernel_size=fh_proc_conf.median_factor)
    gammaT3map = medfilt2d(biasmap, kernel_size=fh_proc_conf.median_factor)
    logging.info("    ...done.")

    return (
        heightmap,
        extinctionmap,
        ratiomap,
        gammaT1map,
        gammaT2map,
        gammaT3map,
        biasmap,
        vertical_wavenumber,
        rg_vec_subs,
        az_vec_subs,
        subs_F_r,
        subs_F_a,
        MBMP_correlation,
    )


def heigths_masking_and_merging(data_equi7_fnames, mask_equi7_fnames, stacks_to_merge_dict):
    # for each stack:
    # 1) check if it is alone, of if there is a couple ASC+DES stacks
    # 2) if alone, gust apply the mask
    # 3) if it is not alone, merge it togheter with its companion data, also applying the mask

    # the asc_des_string is formatted as it follow:
    # XXX:SELF,YYY:STACK_ID
    # where:
    #   XXX can be 'ASC' or 'DES'
    #   YYY can be 'ASC' or 'DES' (if XXX='ASC', than YYY='DES' and vicecersa )
    #   'SELF' is a constant string: the asc_des_string is contained in a dictionary, 'SELF' is the dictionary key
    #   STACK_ID can be 'N.A' or can be the name of a stack_id

    # each input in data_equi7_fnames is a tiff with two layers
    # first later is data
    data_layer_index = 1
    # second layer is quality
    quality_layer_index = 2

    temp_path = next(iter(data_equi7_fnames.values()))[0]
    idx = temp_path.find("equi7")
    merging_folder = os.path.join(temp_path[: idx + len("equi7")], "merged")

    merged_data_fnames = []
    for (current_merging_id, unique_stack_ids_to_merge_list) in stacks_to_merge_dict.items():

        # prepare the empty dictionary for equi7 tile names
        equi7_tile_names = []

        # retrive the names of all the used "EQUI7 tiles"
        for curr_equi7_name in data_equi7_fnames[unique_stack_ids_to_merge_list[0]]:
            curr_equi7_tile_name = os.path.basename(os.path.dirname(curr_equi7_name))
            if not curr_equi7_tile_name in equi7_tile_names:
                equi7_tile_names.append(curr_equi7_tile_name)

        first_stack_data_fname = data_equi7_fnames[unique_stack_ids_to_merge_list[0]][0]
        data_tiff_name_in = os.path.basename(first_stack_data_fname)

        data_tiff_name_out = data_tiff_name_in.replace(unique_stack_ids_to_merge_list[0], current_merging_id)

        p1 = os.path.dirname(os.path.dirname(first_stack_data_fname))
        equi7_subgridname = os.path.basename(p1)

        for equi7_tile_name in equi7_tile_names:

            out_data_fname = os.path.join(
                merging_folder, current_merging_id, equi7_subgridname, equi7_tile_name, data_tiff_name_out,
            )

            os.makedirs(os.path.dirname(out_data_fname))

            for idx, unique_stack_id in enumerate(unique_stack_ids_to_merge_list):
                # input and output paths

                curr_data_fname = [name for name in data_equi7_fnames[unique_stack_id] if equi7_tile_name in name][0]
                curr_mask_fname = [name for name in mask_equi7_fnames[unique_stack_id] if equi7_tile_name in name][0]

                # Load DATA
                data_driver = gdal.Open(curr_data_fname, GA_ReadOnly)
                data_curr = data_driver.GetRasterBand(data_layer_index).ReadAsArray()
                projection = data_driver.GetProjection()
                geotransform = data_driver.GetGeoTransform()
                # Load quality layer
                quality_data_curr = data_driver.GetRasterBand(quality_layer_index).ReadAsArray()
                data_driver = None

                # Load MASK
                mask_driver = gdal.Open(curr_mask_fname, GA_ReadOnly)
                mask_curr = (mask_driver.GetRasterBand(1).ReadAsArray()).astype("bool")
                mask_driver = None

                if idx == 0:
                    FH_out_data = data_curr
                    quality_out_data = quality_data_curr

                    FH_out_data[np.logical_not(mask_curr)] = np.nan
                    quality_out_data[np.logical_not(mask_curr)] = np.nan

                    mask_prev = mask_curr

                else:

                    data_prev = FH_out_data
                    quality_data_prev = quality_out_data
                    del FH_out_data, quality_out_data

                    # 1) mean of the current and previous:
                    data_out_curr = (data_curr + data_prev) / 2
                    quality_data_out_curr = (quality_data_curr + quality_data_prev) / 2

                    # 2) where current only is valid use current values and viceversa
                    curr_only_valid_idxes = np.logical_and(mask_curr, np.logical_not(mask_prev))
                    prev_only_valid_idxes = np.logical_and(mask_prev, np.logical_not(mask_curr))

                    data_out_curr[curr_only_valid_idxes] = data_curr[curr_only_valid_idxes]
                    data_out_curr[prev_only_valid_idxes] = data_prev[prev_only_valid_idxes]
                    quality_data_out_curr[curr_only_valid_idxes] = quality_data_curr[curr_only_valid_idxes]
                    quality_data_out_curr[prev_only_valid_idxes] = quality_data_prev[prev_only_valid_idxes]

                    # 3) where none are valid, write NAN
                    invalid_idxes = np.logical_and(np.logical_not(mask_prev), np.logical_not(mask_curr))

                    data_out_curr[invalid_idxes] = np.nan
                    quality_data_out_curr[invalid_idxes] = np.nan

                    # 4) keep only common parts:
                    not_common_parts_indexes = np.logical_or(np.isnan(data_curr), np.isnan(data_prev))

                    data_out_curr[not_common_parts_indexes] = np.nan
                    quality_data_out_curr[not_common_parts_indexes] = np.nan

                    FH_out_data = data_out_curr
                    quality_out_data = quality_data_out_curr

                    mask_prev = mask_curr

                    del data_out_curr, quality_data_out_curr, data_prev, quality_data_prev

            # save to file:
            Nx, Ny = FH_out_data.shape

            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(out_data_fname, Ny, Nx, 2, gdal.GDT_Float32)
            outdata.SetGeoTransform(geotransform)  ##sets same geotransform as input
            outdata.SetProjection(projection)  ##sets same projection as input
            outdata.GetRasterBand(data_layer_index).WriteArray(FH_out_data)
            outdata.GetRasterBand(quality_layer_index).WriteArray(quality_out_data)
            outdata.FlushCache()  ##saves to disk!!
            outdata = None

        merged_data_fnames.append(out_data_fname)
        logging.info("    ...done.")

    return merged_data_fnames, merging_folder


def get_volume_SB_Pi(Pi, gamma_g, eps=1e-6):
    """
    Extracts the whitened volume component corresponding to the minimum
    Entropy ground solution (rank-2 ground) and the volume coherence
    (gamma_volume) from the given whitened PolInSAr matrix Pi and the
    ground coherence gamma_g.

    In order to avoid numerical errors the minimum eigenvalue of the whitened
    ground is set to eps instead of 0.

    Parameters
    ----------
    Pi : ndarray (3,3)
        Whitened PolInSAR matrix
    gamma_g : complex scalar
        Ground coherence (gamma_ground)
    eps : scalar, optional
        Power minimum ground eigenvalue (whitened). Set it to a small value
        to avoid numerical problems when its equal to 0.

    Returns
    -------
    Tvw : ndarray (3,3)
        Whitened volume component corresponding to a rank-2 ground component
    gamma_v : complex scalar
        Selected volume coherence (gamma_volume)
    """
    # Get center of mass coherence region
    gamma_c = np.trace(Pi) / 3
    # Get whitened volume with gamma_c
    Tvh = (Pi - gamma_g * np.eye(3)) / (gamma_c - gamma_g)
    Tvh = 0.5 * (Tvh + np.conj(Tvh.T))
    # Scale it to max eigenvalue (1.0 - eps)
    l1 = np.max(np.linalg.eigvalsh(Tvh))
    # Get obtained gamma_volume
    gamma_v = gamma_g + (gamma_c - gamma_g) * l1 / (1.0 - eps)
    return Tvh * (1.0 - eps) / l1, gamma_v


def get_Ground_and_Volume_components(Pi, gamma_g=1, eps=1e-6):
    """
    Computes the Ground and Volume components
    corresponding to the minimum Entropy ground solution (rank-2 ground)
    and the relative error.

    Parameters
    ----------
    Pi : ndarray (3,3)
        Whitened PolInSAR matrix
    gamma_g : scalar, optional
        Ground coherence
    eps : scalar, optional
        Power minimum ground eigenvalue (whitened). Set it to a small value
        to avoid numerical problems when its equal to 0.

    Returns
    -------
    gamma_v : complex scalar
        Selected volume coherence (gamma_volume)
    mu : scalar
        minimum ground to volume ratio
    err_fro : scalar
        Frobenius relative squared error performed in the 2 layer modelling
        with respect to original data
    """
    # Get whitened volume component & gamma_volume
    Tvw, gamma_v = get_volume_SB_Pi(Pi, gamma_g, eps)

    # Get whitened ground component
    Tgw = np.eye(3) - Tvw

    # Compute Modelled PolInSAR coherence (within the line)
    Pi_m = gamma_g * Tgw + gamma_v * Tvw

    # Compute Frobenius relative modelling squared error
    err_fro = (np.linalg.norm(Pi - Pi_m) / np.linalg.norm(Pi)) ** 2

    # Minimum Ground-to-volume ratio
    mu = np.min(np.diag(Tgw / Tvw))

    return gamma_v, mu, err_fro
