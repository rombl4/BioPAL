import numpy as np
from numpy import linalg as LA


def do_calc_height_single_bas(PI,kz,offnadir,LUT,param_dict,flags_dict,biases,high_coherence_threshold,num_bas=1,calc_meth="line",error_model_flag=0):
    """this function calculates FH from single baseline

    Parameters
    ----------


    PI: complex array 3x3*3: num_pol*num_pol*num_bas
        PI matrix for 3 baselines

    kz: float
        kz baseline

    offnadir: float
        off nadir angle

    num_bas:  int
        selection of baselines from PI matrix

    biases: float array

    high_coherence_threshold: int
        thresholding the high coherence values and preparing a mask (ambiguous estimation)

    Returns
    -------
    out_dict: dict
        {"height": ,  "mu": }


    Notes
    -----

    Author : Roman Guliaev
    Date : Feb 2022

    """

    # inversion parameters
    kh_vec,  coef_vec_mu,temp_vec = param_dict["kh_vec"],  param_dict["coef_vec_mu"],  param_dict["temp_vec"]


    lut_kh_mu=np.copy(LUT)

    #polarimetric optimization (center of coherence region)
    gammav0_tr = np.trace(PI,axis1=0,axis2=1)/3



    if  flags_dict["coh_opt_flag"]:
        N_ch = PI.shape[0]
        num_baselines = PI.shape[2]
        I = np.eye(N_ch)  # Identity matrix
        gammav0 = np.zeros(num_baselines, dtype=np.complex64)
        for j in np.arange(num_baselines):
            P = np.copy(PI[:, :, j])
            if flags_dict["min_ground_flag"]: P = PI[:, :, j] - I # The furthest point from the ground position is found.
            B = P @ np.conjugate(np.transpose(P))
            w, U = LA.eig(B)
            max_index = np.argmax(w, axis=0)  #find max eigenvalue
            w = U[:, max_index].reshape((1, N_ch))
            gammav0[j] = np.conjugate(w) @ P @ np.transpose(w)
            if flags_dict["min_ground_flag"]: gammav0[j] = gammav0[j] + 1.0


    else:
        gammav0 = np.copy(gammav0_tr)

    if error_model_flag:
        gammav0 = np.copy(gammav0_tr)

    ###range decorrelation compensation
    rg_resolution= 30 # this is the bandwidth limited range resolution: should be recalculated for 6 MHz
    rg_deco=1
    if  ((flags_dict["rg_deco_flag"]) &  error_model_flag==0):  rg_deco = 1 - np.abs(rg_resolution / (2 * np.pi / kz)) * np.cos(offnadir)
    gammav0 = gammav0 / rg_deco
    ###

    ###gamma SNR
    gamma_snr=1
    gammav0 = gammav0 / gamma_snr
    ###

    ###optional: remove values of coherences larger than 1
    ind_more1=np.where(np.abs(gammav0)>1.)
    gammav0[ind_more1]=gammav0[ind_more1]/np.abs(gammav0[ind_more1])
    #gammav0[ind_more1] = 1.
    ###

    ###selection of baseline
    coh = gammav0[num_bas]
    ###
    if flags_dict["calc_meth"] == "mu":
        try:
            ind = np.nanargmin(np.abs(lut_kh_mu - coh))
            ind = np.unravel_index(ind, lut_kh_mu.shape)
            height = kh_vec[ind[0]]
            mu = coef_vec_mu[ind[1]]
            bias = biases[ind[0]]

        except:
            print("error")
            height= np.nan
            mu = np.nan
            bias = np.nan

    if flags_dict["calc_meth"] == "line":

        tangens = np.arctan((1-np.real(coh))/np.imag(coh))
        lut = np.copy(lut_kh_mu[:, 0])
        lut = np.arctan((1 - np.real(lut)) / np.imag(lut))

        try:
            ind = np.nanargmin(np.abs(lut - tangens))
            # ind = np.unravel_index(ind, lut_2d_psf.shape)
            # height[ii] = kh_vec[ind[0]]
            height = kh_vec[ind]
            mu = np.nan
            bias = biases[ind]
        except:
            print("error")
            height = np.nan
            mu = np.nan
            bias =  np.nan

    if flags_dict["calc_meth"] == "coherence":
        ind = np.nanargmin(np.abs(np.abs(lut_kh_mu[:, 0]) - np.abs(coh)))
        height = kh_vec[ind]
        mu = np.nan
        bias = biases[ind]

    if flags_dict["calc_meth"] == "pols2":

        #it is generally a different approach - uses 4d lut inversion - this method is added to match atbd but rather unlikely applicable for the L2 product

        # select 2 pols
        coh1 = PI[0,0,num_bas]
        coh2 = PI[2, 2, num_bas]
        kz1=kz[num_bas]

        ###range decorrelation compensation
        rg_resolution = 30  # this is the bandwidth limited range resolution: should be recalculated for 6 MHz
        rg_deco = 1
        if  ((flags_dict["rg_deco_flag"]) &  error_model_flag==0):  rg_deco = 1 - np.abs(rg_resolution / (2 * np.pi / kz1)) * np.cos(offnadir)
        coh1,coh2 = coh1 / rg_deco,  coh2 / rg_deco
        ###

        ###gamma SNR
        gamma_snr = 1
        coh1,coh2 = coh1 / gamma_snr,  coh2 / gamma_snr
        ###

        ###optional: remove values of coherences larger than 1
        if np.abs(coh1) > 1.: coh1 = coh1 / np.abs(coh1)
        if np.abs(coh1) > 1.: coh2 = coh2 / np.abs(coh2)
        # gammav0[ind_more1] = 1.
        ###

        lut_kh_mu_temp = np.einsum('ab,c->abc', LUT, temp_vec)

        ident = np.ones((len(coef_vec_mu)))

        #lut contains dimensions: kh, mu1,mu2,temp_vec
        cost1 = np.abs(lut_kh_mu_temp - coh1) ** 2
        cost1 = np.einsum('abc,d->abdc', cost1, ident)

        cost2 = np.abs(lut_kh_mu_temp - coh2) ** 2
        cost2 = np.einsum('abc,d->adbc', cost2, ident)

        # find minimum
        ind = np.nanargmin(cost1 + cost2)
        ind = np.unravel_index(ind, cost2.shape)
        height = kh_vec[ind[0]]
        mu = coef_vec_mu[ind[1]]
        mu2 = coef_vec_mu[ind[2]]
        temp1 = temp_vec[ind[3]]
        bias = biases[ind[0]]

    if error_model_flag==0: height = height / np.abs(kz[num_bas])

    #optonal: flagging low coherence values
    # if np.abs(gammav0_tr[num_bas])<.25:
    #     height= np.nan
    #     mu = np.nan

    #high coherence values
    if (np.abs(coh) > high_coherence_threshold):
        bias= np.nan

    out_dict = {"height": height, "mu": mu, "bias": bias}

    return out_dict





def do_calc_height_dual_bas(PI,kz,offnadir,LUT,param_dict,flags_dict,biases,high_coherence_threshold,num_bas1=0,num_bas2=1):
    """this function calculates FH from single baseline

    Parameters
    ----------


    PI: complex array 3x3*3: num_pol*num_pol*num_bas
        PI matrix for 3 baselines

    kz: float
        kz baseline

    offnadir: float
        off nadir angle

    num_bas: selection of baselines from PI matrix



    Returns
    -------
    out_dict: dict
        height, G/V ration, temproral decor for 2 baselines
        {"height": ,  "mu": ,"temp1":,"temp2":}


    Notes
    -----


    Author : Roman Guliaev
    Date : Mar 2022

    """

    #inversion parameters
    kh_vec,coef_vec_mu,temp_vec = param_dict["kh_vec"],param_dict["coef_vec_mu"],param_dict["temp_vec"]

    #LUT with temporal decorrelation
    lut_kh_mu_temp = np.einsum('ab,c->abc', LUT, temp_vec)

    #polarimetric optimization
    #another option is min ground search: to be added as an otion later
    gammav0_tr = np.trace(PI,axis1=0,axis2=1)/3
    gammav0 = np.copy(gammav0_tr)





    if  flags_dict["coh_opt_flag"]:
        # The furthest point from the ground position is found.
        N_ch = PI.shape[0]
        num_baselines = PI.shape[2]

        I = np.eye(N_ch)  # Identity matrix
        gammav0 = np.zeros(num_baselines, dtype=np.complex64)
        for j in np.arange(num_baselines):
            P = np.copy(PI[:, :, j])
            if flags_dict["min_ground_flag"]: P = PI[:, :, j] - I
            B = P @ np.conjugate(np.transpose(P))
            w, U = LA.eig(B)
            max_index = np.argmax(w, axis=0)
            w = U[:, max_index].reshape((1, N_ch))
            gammav0[j] = np.conjugate(w) @ P @ np.transpose(w)
            if flags_dict["min_ground_flag"]: gammav0[j] = gammav0[j] + 1.0


    ### range decorrelation compensation
    rg_resolution = 30  # this is the bandwidth limited range resolution: should be recalculated for 6 MHz
    rg_deco=1
    if  ((flags_dict["rg_deco_flag"])): rg_deco = 1 - np.abs(rg_resolution / (2 * np.pi / kz)) * np.cos(offnadir)

    gammav0 = gammav0 / rg_deco
    ###

    ###gamma SNR
    gamma_snr=1
    gammav0 = gammav0 / gamma_snr
    ###

    ###optional: remove values of coherences larger than 1
    ind_more1 = np.where(np.abs(gammav0) > 1.)
    gammav0[ind_more1] = gammav0[ind_more1] / np.abs(gammav0[ind_more1])
    # gammav0[ind_more1] = 1.
    ###

    #select 2 baselines
    coh1=gammav0[num_bas1]
    coh2=gammav0[num_bas2]
    kz1 = kz[num_bas1]
    kz2 = kz[num_bas2]

    ident = np.ones((len(temp_vec)))

    cost1= np.abs(lut_kh_mu_temp - coh1)** 2
    cost1 = np.einsum('abc,e->abce', cost1, ident)

    cost2= np.abs(lut_kh_mu_temp - coh2)** 2
    cost2 = np.einsum('abc,e->abec', cost2, ident)

    #bringing the two costs to the same height reference (interpolation)
    cost1 = cost1[(kz1/kz2*np.arange(len(kh_vec))).astype(int), :, :, :]

    #find minimum
    ind = np.nanargmin(cost1+cost2)
    ind = np.unravel_index(ind, cost2.shape)
    kh = kh_vec[ind[0]]
    mu = coef_vec_mu[ind[1]]
    temp1 = temp_vec[ind[2]]
    temp2 = temp_vec[ind[3]]
    bias = biases[ind[0]]



    height = kh / np.abs(kz2)

    #flag in the bias
    if flags_dict["high_coherence_threshold_flag"]:
        if ((np.abs(coh1) > high_coherence_threshold ) | (np.abs(coh2) > high_coherence_threshold ) ):
            bias= np.nan

    out_dict = {"height": height, "mu": mu, "t1": temp1, "t2": temp2, "bias": bias}

    return out_dict








def calc_lut_kh_mu(profile,param_dict):

    """this function calculates 2D LUT of complex coherences for each of the values of Kz*H and Mu for the given profile

    Parameters
    ----------


    profile: float array
        profile (all positive real values)

    param_dict: dict


    Returns
    -------
    out_dict: 2d array
        2D LUT


    Notes
    -----

    Author : Roman Guliaev
    Date : Mar 2022

    """

    #parameters for LUT
    kh_vec,  coef_vec_mu,height_vec_norm = param_dict["kh_vec"],  param_dict["coef_vec_mu"], param_dict["height_vec_norm"]

    master_profile = np.copy(profile)

    lut = np.ndarray((len(kh_vec),len(coef_vec_mu)),dtype="complex64")
    for kh in range(len(kh_vec)):
        for mu in range(len(coef_vec_mu)):
            exp_kz = np.exp(1j * kh_vec[kh] * height_vec_norm)  #vector of complex exponentials for a given KzH
            gamma_v = (np.sum(exp_kz * master_profile)) / np.abs(np.sum(master_profile)) #calculating volume only coherence
            lut[kh, mu] = (gamma_v+coef_vec_mu[mu])/(1+coef_vec_mu[mu]) #volumetric coherence with ground component
    return lut





def error_model_kzh(profile,param_dict,flags_dict,decorrelation=.97):


    """this function calculates 1D LUT of biases for each value of KzH

    Parameters
    ----------


    profile: float array
        profile (all positive real values)

    param_dict: dict

    decorrelation: float
        level of assumed non-comensated decorrelation


    Returns
    -------
    height_real: 1d array
        vector of KzH from 0 to 2pi
    height_inverted:   1d array
        vector of inverted KzH assuming decorrelation


    Notes
    -----

    Author : Roman Guliaev
    Date : May 2022

    """

    kh_vec = param_dict["kh_vec"]
    LUT=calc_lut_kh_mu(profile, param_dict)
    height_inverted,height_real=np.zeros(LUT[:,0].shape),np.zeros(LUT[:,0].shape)

    for kh in range(len(kh_vec)):


        #LUT[kh,0] - zero G/V ratio case. can be any though because inversion should be independent of G/V ratio
        #pi_mat - simulated pi matrix
        pi_mat = np.ones((3,3,3)) * LUT[kh,0] * decorrelation
        kh_current = kh_vec[kh]

        #calcluating result for the given decorrelation level
        result_dict = do_calc_height_single_bas(pi_mat, [np.nan,np.nan,np.nan], np.nan, LUT, param_dict, flags_dict,kh_vec*np.nan,1,num_bas=0,calc_meth="line",error_model_flag=1)

        #original and inverted heights
        height_inverted[kh] = result_dict["height"]
        height_real[kh] =  kh_vec[kh]

    return height_real,  height_inverted




def profile_dependent_part(profile,param_dict,flags_dict):


    """this function updates the profile with Gauss fit and calculates profile dependent paramteters: biases vector and coherence threshold for error model

    Parameters

    ----------



    profile:  float vector

        tomo profile


    param_dict: dictionary

    flags_dict: dictionary



    Returns

    -------

    profile: array
        updated Gasuss fitted profile same dimension as original profile
    biases:
        bias for every value of kz*H, is unique for a given profile
    high_coherence_threshold:
        threshold for flagging high coherence values


    Notes

    -----

    Author : Roman Guliaev

    Date : May 2022

    """





    kh_vec = param_dict["kh_vec"]
    ###gauss_fit
    num_el=len(profile)

    if  flags_dict["gauss_flag"]:
        min_el=np.argmin(profile)
        profile=profile[min_el::]
        profile=profile-np.min(profile)
        profile=profile/np.max(profile)
        from scipy.optimize import curve_fit
        xx=np.linspace(0,1,len(profile))
        def Gauss(xx,x0,sigma_g):
            return np.exp(-(xx-x0)**2/(2*sigma_g**2))
        x0, sigma_g=0.5,1

        popt,pcov = curve_fit(Gauss,xx,profile,p0=[x0,sigma_g])

        profile_gauss=np.zeros(num_el)
        profile_orig = np.zeros(num_el)
        profile_gauss[min_el::]=Gauss(xx,*popt)
        profile_orig[min_el::] = profile
        profile=profile_gauss
    ###########




    #error model paramters calculation
    height_real, height_inverted = error_model_kzh(profile, param_dict, flags_dict, decorrelation=.97)

    #find minimum for coherence threshold
    ind=np.argmin(height_inverted)
    high_coherence_threshold=np.abs(np.cos(kh_vec[ind]))

    #biases vector for every kz*H value in percenantage
    biases=np.abs(height_inverted-height_real)/height_real*100
    ###############

    return profile,biases,high_coherence_threshold


def set_param_dict():

    n_sample = 70
    kh_vec_num=int(2 * np.pi/0.05) #sensitivity of 1 m for smaller Kz value of 0.05
    kh_vec = np.linspace(0, 2 * np.pi, kh_vec_num)
    height_vec = np.linspace(0, n_sample - 1, n_sample)
    height_vec_norm = height_vec / (n_sample - 1)

    #G/V ratio, sampled uniformally on complex plane
    mu_space=np.linspace(0.01, .99, 20)
    coef_vec_mu = np.concatenate(([0], (1-mu_space)/mu_space))
    #temp decor sampling: lowest possible is set to 0.3, highest is 1.0
    temp_decor_vec = np.linspace(0.3, 1.0, 20)
    param_dict = { "height_vec_norm": height_vec_norm, "kh_vec": kh_vec,  "coef_vec_mu": coef_vec_mu, "temp_vec": temp_decor_vec}

    return param_dict






