import numpy as np


def do_calc_height_single_bas(PI,kz,offnadir,LUT,param_dict,biases,high_coherence_threshold,num_bas=0,calc_meth="line",rg_deco_flag=1):
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

    # inversion parameters
    kh_vec,  coef_vec_mu = param_dict["kh_vec"],  param_dict["coef_vec_mu"]

    lut_kh_mu=np.copy(LUT)
    #polarimetric optimization (center of coherence region)
    #another option is min ground search: to be added as an otion later
    gammav0_tr = np.trace(PI,axis1=0,axis2=1)/3

    ###range decorrelation compensation
    rg_resolution= 30 # this is the bandwidth limited range resolution: should be recalculated for 6 MHz
    rg_deco=1
    if rg_deco_flag:  rg_deco = 1 - np.abs(rg_resolution / (2 * np.pi / kz)) * np.cos(offnadir)
    gammav0 = np.copy(gammav0_tr)
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
    if calc_meth == "mu":
        try:
            ind = np.nanargmin(np.abs(lut_kh_mu - coh))
            # ind = np.nanargmin(np.abs(np.abs(lut_kh_mu) - np.abs(coh)))
            ind = np.unravel_index(ind, lut_kh_mu.shape)

            meth="abs1"
            if meth=="abs":
                ind[0]=np.nanargmin(np.abs(np.abs(lut_kh_mu[:,0]) - np.abs(coh)))

            hh2 = kh_vec[ind[0]]
            mu2 = coef_vec_mu[ind[1]]
            bias = biases[ind[0]]
            #dist= np.abs(lut_kh_mu[ind] - coh)
        except:
            print("error")
            hh2= np.nan
            mu2 = np.nan
            bias = np.nan

    if calc_meth == "line":

        tangens = np.arctan((1-np.real(coh))/np.imag(coh))
        lut = np.copy(lut_kh_mu[:, 0])
        lut = np.arctan((1 - np.real(lut)) / np.imag(lut))

        try:
            ind = np.nanargmin(np.abs(lut - tangens))
            # ind = np.unravel_index(ind, lut_2d_psf.shape)
            # height[ii] = kh_vec[ind[0]]
            hh2 = kh_vec[ind]
            mu2 = np.nan
            bias = biases[ind]
        except:
            print("error")
            hh2 = np.nan
            mu2 = np.nan
            bias =  np.nan

    if calc_meth == "coherence":
        ind = np.nanargmin(np.abs(np.abs(lut_kh_mu[:, 0]) - np.abs(coh)))
        hh2 = kh_vec[ind]
        mu2 = np.nan
        bias =  np.nan

    hh2 = hh2 / np.abs(kz[num_bas])

    #optonal: flagging low coherence values
    # if np.abs(gammav0_tr[num_bas])<.25:
    #     hh2= np.nan
    #     mu2 = np.nan

    #high coherence values
    #print   (     high_coherence_threshold,np.abs(gammav0_tr)     )
    if ((np.abs(coh) ) > high_coherence_threshold):
        hh2= np.nan
        mu2 = np.nan

    out_dict = {"height": hh2, "mu": mu2, "bias": bias}

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

    #inversion parameters
    k_vec,height_vec,coef_vec_mu,temp_vec = param_dict["k_vec"],param_dict["height_vec"],param_dict["coef_vec_mu"],param_dict["temp_vec"]

    #LUT with temporal decorrelation
    lut_k_h_mu_temp = np.einsum('abc,d->abcd', LUT, temp_vec)


    #polarimetric optimization
    #another option is min ground search: to be added as an otion later
    gammav0_tr = np.trace(PI,axis1=0,axis2=1)/3

    ### range decorrelation compensation
    rg_resolution = 30  # this is the bandwidth limited range resolution: should be recalculated for 6 MHz
    rg_deco = 1 - np.abs(rg_resolution / (2 * np.pi / kz)) * np.cos(offnadir)
    gammav0 = np.copy(gammav0_tr)
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
    coh_com1=gammav0[num_bas1]
    coh_com2=gammav0[num_bas2]
    kz1 = kz[num_bas1]
    kz2 = kz[num_bas2]




    lut4d1[:, :, :, 0] = np.abs(lut_3d_psf - coh_com1[ii])
    for kk in range(lut_3d_psf.shape[2]): lut4d1[:, :, :, kk] = lut4d1[:, :, :, 0]

    lut4d2[:, :, 0, :] = np.abs(lut_3d_psf - coh_com2[ii])
    for kk in range(lut_3d_psf.shape[2]): lut4d2[:, :, kk, :] = lut4d2[:, :, 0, :]

    h_vec1 = np.abs(kh_vec / kz1[ii])
    h_vec2 = np.abs(kh_vec / kz2[ii])

    lut4d1_interm = np.copy(lut4d1)
    for kk in range(lut_3d_psf.shape[0]):
        h_arg = np.argmin(np.abs(h_vec1 - h_vec2[kk]))
        lut4d1[kk, :, :, :] = np.copy(lut4d1_interm[h_arg, :, :, :])

    ind = np.nanargmin(np.abs(lut4d1) ** 2 + np.abs(lut4d2) ** 2)
    # ind = np.nanargmin(np.abs(np.abs(lut_2d_psf) - np.abs(coh)))
    ind = np.unravel_index(ind, lut4d2.shape)
    hh2[ii] = kh_vec[ind[0]]
    mu2[ii] = coef_vec_mu[ind[1]]
    temp1[ii] = temp_vec[ind[2]]
    temp2[ii] = temp_vec[ind[3]]
    # dist[ii]=np.abs(lut_2d_psf[ind] - coh)
    # mumax[ii] = mumax1



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

    hh2 = hh2 / np.abs(kz2)

    ind = np.nanargmin(np.abs(lut_h_mu_temp1_temp2_1) ** 2 + np.abs(lut_h_mu_temp1_temp2_2) ** 2)
    ind = np.unravel_index(ind, lut_h_mu_temp1_temp2_2.shape)
    hh2 = height_vec[ind[0]]
    mu2 = coef_vec_mu[ind[1]]
    temp1 = temp_vec[ind[2]]
    temp2 = temp_vec[ind[3]]

    out_dict = {"height": hh2, "mu": mu2, "t1": temp1, "t2": temp2}

    return out_dict









# def do_calc_height_dual_bas(PI,kz,offnadir,LUT,param_dict,num_bas1=0,num_bas2=1):
#     """this function calculates FH from single baseline
#
#     Parameters
#     ----------
#
#
#     PI: complex array 3x3*3: num_pol*num_pol*num_bas
#         PI matrix for 3 baselines
#
#     kz: float
#         kz baseline
#
#     offnadir: float
#         off nadir angle
#
#     num_bas: selection of baselines from PI matrix
#
#
#
#     Returns
#     -------
#     out_dict: dict
#         height, G/V ration, temproral decor for 2 baselines
#         {"height": ,  "mu": ,"temp1":,"temp2":}
#
#
#     Notes
#     -----
#
#
#     Author : Roman Guliaev
#     Date : Mar 2022
#
#     """
#
#     #inversion parameters
#     k_vec,height_vec,coef_vec_mu,temp_vec = param_dict["k_vec"],param_dict["height_vec"],param_dict["coef_vec_mu"],param_dict["temp_vec"]
#
#     #LUT with temporal decorrelation
#     lut_k_h_mu_temp = np.einsum('abc,d->abcd', LUT, temp_vec)
#
#
#     #polarimetric optimization
#     #another option is min ground search: to be added as an otion later
#     gammav0_tr = np.trace(PI,axis1=0,axis2=1)/3
#
#     ### range decorrelation compensation
#     rg_resolution = 30  # this is the bandwidth limited range resolution: should be recalculated for 6 MHz
#     rg_deco = 1 - np.abs(rg_resolution / (2 * np.pi / kz)) * np.cos(offnadir)
#     gammav0 = np.copy(gammav0_tr)
#     gammav0 = gammav0 / rg_deco
#     ###
#
#     ###gamma SNR
#     gamma_snr=1
#     gammav0 = gammav0 / gamma_snr
#     ###
#
#     ###optional: remove values of coherences larger than 1
#     ind_more1 = np.where(np.abs(gammav0) > 1.)
#     gammav0[ind_more1] = gammav0[ind_more1] / np.abs(gammav0[ind_more1])
#     # gammav0[ind_more1] = 1.
#     ###
#
#     #select 2 baselines
#     coh_com1=gammav0[num_bas1]
#     coh_com2=gammav0[num_bas2]
#     kz1 = kz[num_bas1]
#     kz2 = kz[num_bas2]
#
#
#
#     #select closest kz value for 1st baseline
#     kz11 = np.argmin(np.abs(np.abs(kz1) - k_vec))
#     lut_h_mu_temp = np.copy(lut_k_h_mu_temp[kz11, :, :, :])
#     lut_h_mu_temp1_temp2_1 = np.zeros([lut_h_mu_temp.shape[rr] for rr in [0, 1, 2, 2]], dtype="complex64")
#     #abs distances for 1st baseline
#     lut_h_mu_temp1_temp2_1[:, :, :, 0] = np.abs(lut_h_mu_temp - coh_com1)
#     for kk in range(lut_h_mu_temp1_temp2_1.shape[2]): lut_h_mu_temp1_temp2_1[:, :, :, kk] = lut_h_mu_temp1_temp2_1[ :, :, :, 0]
#
#     # select closest kz value for 2nd baseline
#     kz22 = np.argmin(np.abs(np.abs(kz2) - k_vec))
#     lut_h_mu_temp = np.copy(lut_k_h_mu_temp[kz22, :, :, :])
#     lut_h_mu_temp1_temp2_2 = np.zeros([lut_h_mu_temp.shape[rr] for rr in [0, 1, 2, 2]], dtype="complex64")
#     #abs distances for 2nd baseline
#     lut_h_mu_temp1_temp2_2[:, :, 0, :] = np.abs(lut_h_mu_temp - coh_com2)
#     for kk in range(lut_h_mu_temp1_temp2_1.shape[2]): lut_h_mu_temp1_temp2_2[:, :, kk, :] = lut_h_mu_temp1_temp2_2[    :, :, 0, :]
#
#     #common cost function
#     ind = np.nanargmin(np.abs(lut_h_mu_temp1_temp2_1) ** 2 + np.abs(lut_h_mu_temp1_temp2_2) ** 2)
#     ind = np.unravel_index(ind, lut_h_mu_temp1_temp2_2.shape)
#     hh2 = height_vec[ind[0]]
#     mu2 = coef_vec_mu[ind[1]]
#     temp1 = temp_vec[ind[2]]
#     temp2 = temp_vec[ind[3]]
#
#     out_dict = {"height": hh2, "mu": mu2, "t1": temp1, "t2": temp2}
#
#     return out_dict








def calc_lut_kh_mu(profile,param_dict):

    kh_vec,  coef_vec_mu,height_vec_norm = param_dict["kh_vec"],  param_dict["coef_vec_mu"], param_dict["height_vec_norm"]

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






# def error_model(profile,param_dict,decorrelation=.95):
#
#     k_vec,height_vec,coef_vec_mu,height_vec_norm = param_dict["k_vec"],param_dict["height_vec"],param_dict["coef_vec_mu"],param_dict["height_vec_norm"]
#     lut=calc_lut_k_h_mu(profile, param_dict)
#     LUT=calc_lut_kh_mu(profile, param_dict)
#     height_inverted=np.zeros(lut.shape)
#     height_real = np.zeros(lut.shape)
#
#     for kk in range(len(k_vec)):
#         for hh in range(len(height_vec)):
#             print(kk,hh)
#             for mu in range(len(coef_vec_mu)):
#                 pi_fake = np.ones((3, 3,3)) * lut[kk, hh, mu] * decorrelation
#                 kz_fake = np.asarray([k_vec[kk],k_vec[kk],k_vec[kk]])
#                 offnadir=.7
#                 result_dict = do_calc_height_single_bas(pi_fake, kz_fake, offnadir, LUT, param_dict, num_bas=0,calc_meth="line")
#                 height_inverted[kk,hh,mu] = result_dict["height"]
#                 height_real[kk, hh, mu] =  height_vec[hh]
#
#     return height_real  ,  height_inverted



def error_model_kzh(profile,param_dict,calc_meth="coherence",decorrelation=.97):

    kh_vec = param_dict["kh_vec"]
    LUT=calc_lut_kh_mu(profile, param_dict)
    height_inverted=np.zeros(LUT[:,0].shape)
    height_real = np.zeros(LUT[:,0].shape)

    for kh in range(len(kh_vec)):


        pi_mat = np.ones((3,3,3)) * LUT[kh,0] * decorrelation
        kh_current = kh_vec[kh]
        kz=.1
        kz_vector = np.asarray([kz,kz,kz])

        offnadir=.7
        result_dict = do_calc_height_single_bas(pi_mat, kz_vector, offnadir, LUT, param_dict, kh_vec*0,1,num_bas=0,calc_meth=calc_meth,rg_deco_flag=0)

        height_inverted[kh] = result_dict["height"]*kz
        height_real[kh] =  kh_vec[kh]

    return height_real,  height_inverted




def profile_dependent_part(profile,param_dict):


    """this function updates the profile with Gauss fit and calculates profile dependent paramteters: biases vector and coherence threshold for error model



    Parameters

    ----------





    profile:  float vector

        tomo profile


    param_dict: dictionary




    Returns

    -------

    profile: updated Gasuss fitted profile
    biases: bias for every value of kz*H, is unique for a given profile
    high_coherence_threshold: threshold for flagging high coherence values


    Notes

    -----

    Author : Roman Guliaev

    Date : Feb 2022

    """





    kh_vec = param_dict["kh_vec"]
    ###gauss_fit
    num_el=len(profile)
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
    height_real, height_inverted = error_model_kzh(profile, param_dict, calc_meth="line", decorrelation=.97)

    #find minimum for coherence threshold
    ind=np.argmin(height_inverted)
    high_coherence_threshold=np.abs(np.cos(kh_vec[ind]))

    #biases vector for every kz*H value
    biases=np.abs(height_inverted-height_real)/height_real*100
    ###############

    return profile,biases,high_coherence_threshold


def set_param_dict():

    n_sample = 70

    kh_vec = np.linspace(0, 2 * np.pi, 200)
    height_vec = np.linspace(0, n_sample - 1, n_sample)
    height_vec_norm = height_vec / (n_sample - 1)
    #G/V ratio sampling tbd
    coef_vec_mu = np.concatenate(([0], 10 ** np.linspace(-2, 1, 20)))
    #temp decor sampling tbd
    temp_decor_vec = np.linspace(0.7, 1.0, 7)
    param_dict = { "height_vec_norm": height_vec_norm, "kh_vec": kh_vec,  "coef_vec_mu": coef_vec_mu, "temp_vec": temp_decor_vec}

    return param_dict



"""
import matplotlib.pyplot as plt
kk=12
# plt.plot(height_real[kk,:,0])
# plt.plot(height_inverted[kk,:,0])
plt.figure()
#plt.plot(height_real[kk,:,0],height_inverted[kk,:,0],label="$k_z$="+str(k_vec[kk])[0:5])
plt.plot(height_real[:,0],height_inverted[:,0],label="$k_z$="+str(k_vec[kk])[0:5])
plt.xlim([0,70])
plt.ylim([0,70])
plt.plot([0, 70], [0, 70], color="black",linewidth=3,linestyle="dashed")
axes = plt.gca()
axes.set_xlabel('Reference Height')
axes.set_ylabel('Inverted Height')
plt.legend()
"""

"""
import matplotlib.pyplot as plt
kk=0
plt.figure()
plt.plot(height_real[:],height_inverted[:],label="$k_z$="+str(0)[0:5])
set_lim=2*np.pi
plt.xlim([0,set_lim])
plt.ylim([0,set_lim])
plt.plot([0, set_lim], [0, set_lim], color="black",linewidth=3,linestyle="dashed")
axes = plt.gca()
axes.set_xlabel('Reference Height')
axes.set_ylabel('Inverted Height')
plt.legend()
"""



