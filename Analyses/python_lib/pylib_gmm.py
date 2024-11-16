#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:02:51 2024

@author: glavrent
"""
import os 
import numpy as np
import pandas as pd

#file directory
dir_file = os.path.dirname(os.path.realpath(__file__))

#%% Ergodic GMM
### ------------------------------------------
#original short-distance saturation
def gmm_eas(mag, ztor, sof, rrup, vs30, z1p0, reg=['ALL'],
            regions=['ALL'],
            c1=[-3.5], c2=1.27, c3=2.8, c4=-1.86, 
            c5=7.6, c6=0.45, c7=[-0.004], c8=[-1.11], c9=0.0038,
            c10a=-0.2, c10b=0.2, 
            c11=[0.19, 0.12, 0.15, 0.15], z1p0_vs30_brk=[200, 300, 500],
            cn=1.60, cmag=6.75, chm=3.838, ztor_max=20,
            s1=0.50, s2=0.40, s3=0.002, s4=0.42, s5=0.40, s6=0.45,
            s1mag=4.0, s2mag=6.0, s5mag=4.0, s6mag=6.0):
    
    #number of ground motions
    n_gm = len(mag)
    assert(len(ztor) == n_gm),'Error. Invalid number of ztor values'
    assert(len(sof)  == n_gm),'Error. Invalid number of sof values'
    assert(len(rrup) == n_gm),'Error. Invalid number of rrup values'
    assert(len(vs30) == n_gm),'Error. Invalid number of vs30 values'
    assert(len(reg)  == n_gm),'Error. Invalid number of reg values'
    
    #regionalized median scaling
    flag_c1_reg = False if type(c1) is float or type(c1) is np.double else True
    flag_c3_reg = False if type(c3) is float or type(c3) is np.double else True
    flag_c7_reg = False if type(c7) is float or type(c7) is np.double else True
    flag_c8_reg = False if type(c8) is float or type(c8) is np.double else True
    #regionalized aleatory variability
    flag_s1_reg = False if type(s1) is float or type(s1) is np.double else True
    flag_s2_reg = False if type(s2) is float or type(s2) is np.double else True
    flag_s3_reg = False if type(s3) is float or type(s3) is np.double else True
    flag_s4_reg = False if type(s4) is float or type(s4) is np.double else True
    flag_s5_reg = False if type(s5) is float or type(s5) is np.double else True
    flag_s6_reg = False if type(s6) is float or type(s6) is np.double else True
   
    #add z1.0 break vs30
    z1p0_vs30_brk = np.insert(z1p0_vs30_brk, 0, 0)
    
    #region id
    reg_id = np.array([np.where(r==regions)[0] for r in reg]).flatten()
    
    #Intercept
    f_intr = np.ones(mag.shape)
    
    #Source Scaling Terms
    f_src_lin  = mag 
    f_src_fc   = 1.0 / cn * np.log(1 + np.exp(cn * (cmag - mag)))
    f_src_ztor = np.minimum(ztor, ztor_max)
    #sof scaling
    f_src_n = (sof==-1).astype(float)
    f_src_r = (sof==+1).astype(float)
    
    #Path Scaling Terms
    #geometrical spreading
    f_path_gs    = np.log(rrup + c5 * np.cosh(c6 * np.maximum(mag - chm, 0)))
    f_path_gs   -= np.log(np.sqrt(rrup**2 + 50**2))
    #anelastic attenuation
    f_path_atten = rrup  
    #response adjustment
    f_path_adj = -0.5 * np.log(np.sqrt(rrup**2 + 50**2))
    
    #Site Scaling Terms
    #vs30 scaling
    f_site_vs30 = np.log(np.minimum(vs30, 1000.) / 1000.)
    #z1.0 scaling
    z1p0_ref = calcBA19z1(vs30)
    f_site_z1p0 = np.log((np.minimum(z1p0, 2) + 0.01) / (z1p0_ref + 0.01))
    #index z1.0 break
    j_brk_z1p0 = np.array([sum(z1p0_vs30_brk < v)-1 for v in vs30])
    f_site_z1p0_mat = np.zeros( (n_gm, len(z1p0_vs30_brk)) )
    for j, j_b in enumerate(j_brk_z1p0):
        f_site_z1p0_mat[j,j_b] = f_site_z1p0[j]
    
    #Median Ground Motion
    #source scaling
    f_src   = (c1[reg_id] if flag_c1_reg else c1) * f_intr
    f_src  += c2 * f_src_lin 
    f_src  += (c2 - (c3[reg_id] if flag_c3_reg else c3)) * f_src_fc 
    f_src  += c9 * f_src_ztor
    f_src  += c10a * f_src_n + c10b * f_src_r
    #path scaling
    f_path  = c4 * f_path_gs
    f_path += (c7[reg_id] if flag_c7_reg else c7) * f_path_atten + f_path_adj
    #site scalign
    f_site  = (c8[reg_id] if flag_c8_reg else c8) * f_site_vs30
    f_site += f_site_z1p0_mat @ np.array([c11]).flatten()
    #ground motion
    f_med = f_src + f_path + f_site
    
    #Aleatory variability
    s1_array = s1[reg_id] if flag_s1_reg else np.full(n_gm, s1) 
    s2_array = s2[reg_id] if flag_s2_reg else np.full(n_gm, s2)
    s3_array = s3[reg_id] if flag_s3_reg else np.full(n_gm, s3)
    s4_array = s4[reg_id] if flag_s4_reg else np.full(n_gm, s4)
    s5_array = s5[reg_id] if flag_s5_reg else np.full(n_gm, s5)
    s6_array = s6[reg_id] if flag_s6_reg else np.full(n_gm, s6)
    
    #tau
    tau  = np.array([np.interp(m, [s1mag, s2mag], [s1, s2], left=s1, right=s2) 
                     for m, s1, s2 in zip(mag, s1_array, s2_array)])
    #tauP
    tauP = s3_array 
    #phiS
    phiS = s4_array
    #phi
    phi  = np.array([np.interp(m, [s5mag, s6mag], [s5, s6], left=s5, right=s6)
                     for m, s5, s6 in zip(mag, s5_array, s6_array)])
        
    return f_med, f_src, f_path, f_site, tau, tauP, phiS, phi

#updated short-distance saturation
def gmm_eas_upd(mag, ztor, sof, rrup, vs30, z1p0, reg=['ALL'],
                regions=['ALL'],
                c1=[-3.5], c2=1.27, c3=2.8, c4=-1.86, 
                c5=7.6, c6=0.45, c7=[-0.004], c8=[-1.11], c9=0.0038,
                c10a=-0.2, c10b=0.2, 
                c11=[0.19, 0.12, 0.15, 0.15], z1p0_vs30_brk=[200, 300, 500],
                cn=1.60, cmag=6.75, chm=3.838, ztor_max=20,
                s1=0.50, s2=0.40, s3=0.002, s4=0.42, s5=0.40, s6=0.45,
                s1mag=4.0, s2mag=6.0, s5mag=4.0, s6mag=6.0):
    
    #number of ground motions
    n_gm = len(mag)
    assert(len(ztor) == n_gm),'Error. Invalid number of ztor values'
    assert(len(sof)  == n_gm),'Error. Invalid number of sof values'
    assert(len(rrup) == n_gm),'Error. Invalid number of rrup values'
    assert(len(vs30) == n_gm),'Error. Invalid number of vs30 values'
    assert(len(reg)  == n_gm),'Error. Invalid number of reg values'
    
    #convert regions to numpy array
    regions = list(regions) if isinstance(regions, type({}.keys())) else regions
    regions = np.array(regions) if type(regions) is list else regions
    
    #regionalized median scaling
    flag_c1_reg = False if type(c1) is float or type(c1) is np.double else True
    flag_c3_reg = False if type(c3) is float or type(c3) is np.double else True
    flag_c7_reg = False if type(c7) is float or type(c7) is np.double else True
    flag_c8_reg = False if type(c8) is float or type(c8) is np.double else True
    #regionalized aleatory variability
    flag_s1_reg = False if type(s1) is float or type(s1) is np.double else True
    flag_s2_reg = False if type(s2) is float or type(s2) is np.double else True
    flag_s3_reg = False if type(s3) is float or type(s3) is np.double else True
    flag_s4_reg = False if type(s4) is float or type(s4) is np.double else True
    flag_s5_reg = False if type(s5) is float or type(s5) is np.double else True
    flag_s6_reg = False if type(s6) is float or type(s6) is np.double else True
   
    #add z1.0 break vs30
    z1p0_vs30_brk = np.insert(z1p0_vs30_brk, 0, 0)
    
    #region id
    reg_id = np.array([np.where(r==regions)[0] for r in reg]).flatten()
    
    #Intercept
    f_intr = np.ones(mag.shape)
    
    #Source Scaling Terms
    f_src_lin  = mag 
    f_src_fc   = 1.0 / cn * np.log(1 + np.exp(cn * (cmag - mag)))
    f_src_ztor = np.minimum(ztor, ztor_max)
    #sof scaling
    f_src_n = (sof==-1).astype(float)
    f_src_r = (sof==+1).astype(float)
    
    #Path Scaling Terms
    #geometrical spreading
    f_path_gs    = np.log(rrup + c5 * np.exp(c6 * np.maximum(mag - chm, 0)))
    f_path_gs   -= np.log(np.sqrt(rrup**2 + 50**2))
    #anelastic attenuation
    f_path_atten = rrup  
    #response adjustment
    f_path_adj = -0.5 * np.log(np.sqrt(rrup**2 + 50**2))
    
    #Site Scaling Terms
    #vs30 scaling
    f_site_vs30 = np.log(np.minimum(vs30, 1000.) / 1000.)
    #z1.0 scaling
    z1p0_ref = calcBA19z1(vs30)
    f_site_z1p0 = np.log((np.minimum(z1p0, 2) + 0.01) / (z1p0_ref + 0.01))
    #index z1.0 break
    j_brk_z1p0 = np.array([sum(z1p0_vs30_brk < v)-1 for v in vs30])
    f_site_z1p0_mat = np.zeros( (n_gm, len(z1p0_vs30_brk)) )
    for j, j_b in enumerate(j_brk_z1p0):
        f_site_z1p0_mat[j,j_b] = f_site_z1p0[j]
    
    #Median Ground Motion
    #source scaling
    f_src   = (c1[reg_id] if flag_c1_reg else c1) * f_intr
    f_src  += c2 * f_src_lin 
    f_src  += (c2 - (c3[reg_id] if flag_c3_reg else c3)) * f_src_fc 
    f_src  += c9 * f_src_ztor
    f_src  += c10a * f_src_n + c10b * f_src_r
    #path scaling
    f_path  = c4 * f_path_gs
    f_path += (c7[reg_id] if flag_c7_reg else c7) * f_path_atten + f_path_adj
    #site scalign
    f_site  = (c8[reg_id] if flag_c8_reg else c8) * f_site_vs30
    f_site += f_site_z1p0_mat @ np.array([c11]).flatten()
    #ground motion
    f_med = f_src + f_path + f_site
    
    #Aleatory variability
    s1_array = s1[reg_id] if flag_s1_reg else np.full(n_gm, s1) 
    s2_array = s2[reg_id] if flag_s2_reg else np.full(n_gm, s2)
    s3_array = s3[reg_id] if flag_s3_reg else np.full(n_gm, s3)
    s4_array = s4[reg_id] if flag_s4_reg else np.full(n_gm, s4)
    s5_array = s5[reg_id] if flag_s5_reg else np.full(n_gm, s5)
    s6_array = s6[reg_id] if flag_s6_reg else np.full(n_gm, s6)
    
    #tau
    tau  = np.array([np.interp(m, [s1mag, s2mag], [s1, s2], left=s1, right=s2) 
                     for m, s1, s2 in zip(mag, s1_array, s2_array)])
    #tauP
    tauP = s3_array 
    #phiS
    phiS = s4_array
    #phi
    phi  = np.array([np.interp(m, [s5mag, s6mag], [s5, s6], left=s5, right=s6)
                     for m, s5, s6 in zip(mag, s5_array, s6_array)])
        
    return f_med, f_src, f_path, f_site, tau, tauP, phiS, phi

#%% Z1.0 scaling
### ------------------------------------------
def calcBA19z1(vs30: float) -> float:
    #coefficients
    pwr = 4
    v_ref = 610
    slope = -7.67 / pwr
    
    #calc depth to 1000 m/sec
    z1 = np.exp(slope * np.log((vs30**pwr + v_ref**pwr) / (1360.0**pwr + v_ref**pwr))) / 1000.

    #return depth to 1000 m/sec
    return z1


#%% Hangign wall scaling
### ------------------------------------------
def scalingHW(mag, w, dip, z_tor, 
              r_jb, r_rup, r_x, r_y0):
    
    #distance metrics
    r_1 = w * np.cos(np.deg2rad(dip))
    r_2 = 3.0 * r_1
    r_y1 = r_x * np.tan(np.deg2rad(20))
    
    #mag tapering coefficient
    a_2hw = 0.2
    #distance tapering coefficients
    h_1 = 0.25
    h_2 = 1.50
    h_3 =-0.75
    
    #tapering terms
    #dip tapering
    t_1  = np.maximum( (90.-dip)/45., 60./45. )
    #mag tapering
    t_2  = 1. + a_2hw * np.maximum(mag-6.5,-1.)
    t_2 -= (1. - a_2hw) * np.minimum( np.maximum((mag-6.5),-1.), .0)**2
    #normal distance tapering
    t_3  = np.select([r_x < r_1, r_x < r_2],
                     [h_1 + h_2 * (r_x/r_1) + h_3 * (r_x/r_1)**2, 
                      1. - (r_x - r_1)/(r_2 - r_1)], 0.)
    #depth to top of rupture tapering
    t_4 = 1. - np.minimum(z_tor, 10)**2/100
    #parallel distance tapering
    if np.isnan(r_y0).all():
        t_5 = np.minimum( np.maximum( 1 - r_jb/30, 1. ), 0.)
    else:
        t_5 = np.select([r_y0-r_y1<0, r_y0-r_y1<5],
                        [1., 1.-(r_y0-r_y1)/5.], 0.)
        
    #compute hanging wall scaling
    f_hw = t_1 * t_2 * t_3 * t_4 * t_5
    
    return f_hw


#%% Non-linear scaling
### ------------------------------------------
fn_coeffs_nl = '../../Raw_files/model_coeffs/EAS_N2_2_MODEL_COEFFICIENTS.csv'
df_coeffs_nl = pd.read_csv(os.path.join(dir_file,fn_coeffs_nl))

def scalingNL(Vs30, freq, Ir):
    
    #reference shear wave-velocity
    Vsref = 800.
    
    #scaling coeffs
    f3 = np.interp(np.log(freq), np.log(df_coeffs_nl.freq), df_coeffs_nl.f3)
    f4 = np.interp(np.log(freq), np.log(df_coeffs_nl.freq), df_coeffs_nl.f4)
    f5 = np.interp(np.log(freq), np.log(df_coeffs_nl.freq), df_coeffs_nl.f5)
    f2 = f4 * ( np.exp(f5*(np.minimum(Vs30, Vsref)-360.)) - np.exp(f5*(Vsref-360.)) )

    #non-linear amplification
    f_nl = f2 * np.log( (Ir+f3)/f3 )
        
    return f_nl


#%% SOF
### ------------------------------------------
def BA07sof(p_axis_plunge, t_axis_plunge):
    '''Boore and Atkinson 2007'''
    
    if   p_axis_plunge <= 40 and t_axis_plunge <= 40: sof = 'strike-slip'
    elif p_axis_plunge <= 40 and t_axis_plunge >  40: sof = 'reverse'
    elif p_axis_plunge >  40 and t_axis_plunge <= 40: sof = 'normal'
    else: sof = 'unclassified'

    return sof

def Z92sof(p_axis_plunge, t_axis_plunge):
    '''Zoback 1992'''

    if   p_axis_plunge <  40 and t_axis_plunge <= 20: sof = 'strike-slip'
    elif p_axis_plunge <= 20 and t_axis_plunge <  40: sof = 'strike-slip'
    elif p_axis_plunge <= 35 and t_axis_plunge >= 52: sof = 'reverse'
    elif p_axis_plunge >= 52 and t_axis_plunge <= 35: sof = 'normal'
    elif p_axis_plunge <= 20 and t_axis_plunge >  40 and t_axis_plunge <  52: sof = 'reverse-oblique'
    elif p_axis_plunge >  40 and p_axis_plunge <  52 and t_axis_plunge <= 20: sof = 'normal-oblique'
    else: sof = 'unclassified'
    
    return sof

def W94sof(rake_angle):
    '''Wells and Coppersmith 1994'''

    if   abs(rake_angle) < 45 or abs(rake_angle) > 135: sof = 'strike-slip'
    elif rake_angle <=  135 and rake_angle >=  45: sof = 'reverse'
    elif rake_angle >= -135 and rake_angle <= -45: sof = 'normal'
    else: sof  = 'unclassified'

    return sof