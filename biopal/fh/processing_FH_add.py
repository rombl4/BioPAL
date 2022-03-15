import numpy as np


def do_calc_height_single_bas(PI,kz,offnadir,LUT,param_dict,num_bas=0):
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
        {"height": ,  "mu": }


    Notes
    -----

    Author : Roman Guliaev
    Date : Feb 2022

    """

    lut_kh_mu=np.copy(LUT)
    #polarimetric optimization (center of coherence region)
    #another option is min ground search: to be added as an otion later
    gammav0_tr = np.trace(PI,axis1=0,axis2=1)

    ###range decorrelation compensation
    rg_resolution= 30 # this is the bandwidth limited range resolution: should be recalculated for 6 MHz
    rg_deco = 1 - np.abs(rg_resolution / (2 * np.pi / kz)) * np.cos(offnadir)
    gammav0 = np.copy(gammav0_tr)
    gammav0 = gammav0 / rg_deco
    ###

    ###optional: remove values of coherences larger than 1
    ind_more1=np.where(np.abs(gammav0)>1.)
    gammav0[ind_more1]=gammav0[ind_more1]/np.abs(gammav0[ind_more1])
    #gammav0[ind_more1] = 1.
    ###

    ###selection of baseline
    coh = gammav0[num_bas]
    ###

    try:
        ind = np.nanargmin(np.abs(lut_kh_mu - coh))
        # ind = np.nanargmin(np.abs(np.abs(lut_kh_mu) - np.abs(coh)))
        ind = np.unravel_index(ind, lut_kh_mu.shape)
        hh2 = kh_vec[ind[0]]
        mu2 = coef_vec_mu[ind[1]]
        dist= np.abs(lut_kh_mu[ind] - coh)
    except:
        print("error")
        hh2= np.nan
        mu2 = np.nan

    hh2 = hh2 / np.abs(kz[num_bas])

    #optonal: flagging low coherence values
    # if np.abs(gammav0_tr[num_bas])<.25:
    #     hh2= np.nan
    #     mu2 = np.nan

    out_dict = {"height": hh2, "mu": mu2}

    return out_dict









def do_calc_height_dual_bas(PI,kz,offnadir,LUT,param_dict,num_bas1=0,num_bas2=1):
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


    #load precalculated LUT
    #lut_k_h_mu = np.load("/data/largeHome/guli_ro/python/fsar/regions/lope/test/lut_k_h_mu_lambda_res_raw[2, 2]res_[1, 1]filt_[17, 4]P.npy")
    k_vec,height_vec,coef_vec_mu,temp_vec = param_dict["k_vec"],param_dict["height_vec"],param_dict["coef_vec_mu"],param_dict["temp_vec"]
    #LUT with temporal decorrelation
    lut_k_h_mu_temp = np.einsum('abc,d->abcd', LUT, temp_vec)


    #polarimetric optimization
    #another option is min ground search: to be added as an otion later
    gammav0_tr = np.trace(PI,axis1=0,axis2=1)

    # range decorrelation compensation
    rg_resolution = 30  # this is the bandwidth limited range resolution: should be recalculated for 6 MHz
    rg_deco = 1 - np.abs(rg_resolution / (2 * np.pi / kz)) * np.cos(offnadir)
    gammav0 = np.copy(gammav0_tr)
    gammav0 = gammav0 / rg_deco

    # optional: remove values of coherences larger than 1
    ind_more1 = np.where(np.abs(gammav0) > 1.)
    gammav0[ind_more1] = gammav0[ind_more1] / np.abs(gammav0[ind_more1])
    # gammav0[ind_more1] = 1.

    #select 2 baselines
    coh_com1=gammav0[num_bas1]
    coh_com2=gammav0[num_bas2]
    kz1 = kz[num_bas1]
    kz2 = kz[num_bas2]



    #select closest kz value for 1st baseline
    kz11 = np.argmin(np.abs(np.abs(kz1) - k_vec))
    lut_h_mu_temp = np.copy(lut_k_h_mu_temp[kz11, :, :, :])
    lut_h_mu_temp1_temp2_1 = np.zeros([lut_h_mu_temp.shape[rr] for rr in [0, 1, 2, 2]], dtype="complex64")
    #abs distances for 1st baseline
    lut_h_mu_temp1_temp2_1[:, :, :, 0] = np.abs(lut_h_mu_temp - coh_com1)
    for kk in range(lut_h_mu_temp1_temp2_1.shape[2]): lut_h_mu_temp1_temp2_1[:, :, :, kk] = lut_h_mu_temp1_temp2_1[ :, :, :, 0]

    # select closest kz value for 2nd baseline
    kz22 = np.argmin(np.abs(np.abs(kz2) - k_vec))
    lut_h_mu_temp = np.copy(lut_k_h_mu_temp[kz22, :, :, :])
    lut_h_mu_temp1_temp2_2 = np.zeros([lut_h_mu_temp.shape[rr] for rr in [0, 1, 2, 2]], dtype="complex64")
    #abs distances for 2nd baseline
    lut_h_mu_temp1_temp2_2[:, :, 0, :] = np.abs(lut_h_mu_temp - coh_com2)
    for kk in range(lut_h_mu_temp1_temp2_1.shape[2]): lut_h_mu_temp1_temp2_2[:, :, kk, :] = lut_h_mu_temp1_temp2_2[    :, :, 0, :]

    #common cost function
    ind = np.nanargmin(np.abs(lut_h_mu_temp1_temp2_1) ** 2 + np.abs(lut_h_mu_temp1_temp2_2) ** 2)
    ind = np.unravel_index(ind, lut_h_mu_temp1_temp2_2.shape)
    hh2 = height_vec[ind[0]]
    mu2 = coef_vec_mu[ind[1]]
    temp1 = temp_vec[ind[2]]
    temp2 = temp_vec[ind[3]]

    out_dict = {"height": hh2, "mu": mu2, "t1": temp1, "t2": temp2}

    return out_dict








def calc_lut_kh_mu(profile,param_dict):
    kh_vec, height_vec, coef_vec_mu,height_vec_norm = param_dict["kh_vec"], param_dict["height_vec"], param_dict["coef_vec_mu"], param_dict["height_vec_norm"]

    lut = np.ndarray((len(kh_vec),len(coef_vec_mu)),dtype="complex64")
    for kh in range(len(kh_vec)):
        for mu in range(len(coef_vec_mu)):
            master_profile = np.copy(profile)
            master_profile[0] = master_profile[0]+np.sum(master_profile)*coef_vec_mu[mu]
            kh_current = kh_vec[kh]
            exp_kz = np.exp(1j * kh_current * height_vec_norm)
            lut[kh,mu] = (np.sum(exp_kz * master_profile)) / np.abs(np.sum(master_profile))
    return lut



def calc_lut_k_h_mu(profile,param_dict):

    k_vec,height_vec,coef_vec_mu,height_vec_norm = param_dict["k_vec"],param_dict["height_vec"],param_dict["coef_vec_mu"],param_dict["height_vec_norm"]
    lut = np.ndarray((len(k_vec),len(height_vec),len(coef_vec_mu)),dtype="complex64")

    for kk in range(len(k_vec)):
        for hh in range(len(height_vec)):
            print(kk,hh)
            for mu in range(len(coef_vec_mu)):
                master_profile=np.copy(profile)
                master_profile[0] = master_profile[0]+np.sum(master_profile)*coef_vec_mu[mu]
                kh_current = k_vec[kk]*height_vec[hh]
                exp_kz = np.exp(1j * kh_current * height_vec_norm)
                lut[kk,hh,mu] = (np.sum(exp_kz * master_profile)) / np.abs(np.sum(master_profile))
    return lut










