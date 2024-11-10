#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 08:22:49 2024

@author: glavrent
"""
#load libraries
#general
import os
import sys
import pathlib
from warnings import simplefilter
#string libraries
import re
#arithmetic libraries
import numpy as np
import numpy.matlib as mlib
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
#user functions
def calcBA19z1(vs30: float) -> float:
    #coefficients
    pwr = 4
    v_ref = 610
    slope = -7.67 / pwr
    
    #calc depth to 1000 m/sec
    z1 = np.exp(slope * np.log((vs30**pwr + v_ref**pwr) / (1360.0**pwr + v_ref**pwr))) / 1000.

    #return depth to 1000 m/sec
    return z1

    

def gmm_eas_med(mag, ztor, sof, rrup, vs30, z1p0, 
                c5=7.5818, c6=0.45,
                cn=1.60, cmag=6.75, chm=3.838, 
                ztor_max=20, vs30_brk_z1p0=[]):
    
    #number of ground motions
    n_gm = len(mag)
    
    #add z1.0 break vs30
    vs30_brk_z1p0 = np.insert(vs30_brk_z1p0, 0, 0)
        
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
    f_site_z1p0 =  np.log((np.minimum(z1p0, 2) + 0.01) / (z1p0_ref + 0.01))
    #index z1.0 break
    j_brk_z1p0 = np.array([sum(vs30_brk_z1p0 < v)-1 for v in vs30])
    f_site_z1p0_mat = np.zeros( (n_gm, len(vs30_brk_z1p0)) )
    for j, j_b in enumerate(j_brk_z1p0):
        f_site_z1p0_mat[j,j_b] = f_site_z1p0[j]
    
    #organize terms
    f_src  = (f_src_lin, f_src_fc, f_src_ztor, f_src_r, f_src_n)
    f_path = (f_path_gs, f_path_atten)
    f_site = (f_site_vs30, f_site_z1p0, f_site_z1p0_mat)
    
    return f_intr, f_src, f_path, f_site, f_path_adj


def gmm_eas_sig(mag, s1=0.55, s2=0.42, s3=0.02, s4=0.35, s5=0.40, s6=0.40,
                s1mag=4., s2mag=6., s4mag=4., s5mag=6., s6mag=4.):

    #number of ground motions
    n_gm = len(mag)
    
    #tau
    tau  = np.array([np.interp(m, [s1mag, s2mag], [s1, s2], left=s1, right=s2) for m in mag])
    #tauP
    tauP = np.full(n_gm, s3)
    #phiS
    phiS = np.full(n_gm, s4)
    #phi
    phi  = np.array([np.interp(m, [s5mag, s6mag], [s5, s6], left=s5, right=s6) for m in mag])
    
    return tau, tauP, phiS, phi

#suppress warning
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#%% Define Variables
### ======================================
#heteroscedasticity option
flag_hetero_sd = True
#regionaliztion option
flag_reg_med   = True
flag_reg_aleat = True

#regionaliztion of intercpet
reg_intrcp_scl = {'PW': 1.00,
                  'AK': 1.05,
                  'SEE':1.10, 
                  'CH': 0.95, 
                  'JPN':0.80,  
                  'NZ': 1.15,  
                  'TWN':0.90}

#regionaliztion of anelastic attenuation
reg_atten_scl =  {'PW': 1.00,
                  'AK': 1.05,
                  'SEE':1.10, 
                  'CH': 0.95, 
                  'JPN':0.80,  
                  'NZ': 1.15,  
                  'TWN':0.90}

#regionalization of vs30
reg_vs30_scl =   {'PW': 1.00,
                  'AK': 1.05,
                  'SEE':1.10, 
                  'CH': 0.95, 
                  'JPN':0.80,  
                  'NZ': 1.15,  
                  'TWN':0.90}

#regionaliztion of aleatory variability
reg_tauP_scl =  {'PW': 1.00,
                 'AK': 1.05,
                 'SEE':1.10, 
                 'CH': 0.95, 
                 'JPN':0.80,  
                 'NZ': 1.15,  
                 'TWN':0.90}

reg_phi_scl =  {'PW': 1.00,
                'AK': 1.05,
                'SEE':1.10, 
                'CH': 0.95, 
                'JPN':0.80,  
                'NZ': 1.15,  
                'TWN':0.90}

reg_phiS_scl = {'PW':1.00,
                'AK': 1.05,
                'SEE':1.10, 
                'CH': 0.95, 
                'JPN':0.80,  
                'NZ': 1.15,  
                'TWN':0.90}

#correlation lengths
phiS_ell = 10

#number of random realizations
n_realiz = 10

#flatfiel filename
fn_fltfile = '../../../Data/gmm_ergodic/dataset/fltfile_nga3_20240704_rev1.csv'
fn_fltfile = '../../../Data/gmm_ergodic/dataset/fltfile_nga3_20240920_all.csv'

#BA18 coefficients
fn_ba18coeffs = '../../../Raw_files/BA18coefs_mod.csv'

#output directory
dir_out = '../../../Data/gmm_ergodic/verification/dataset/original_gs/'
dir_out += 'heteroscedastic/' if flag_hetero_sd else 'homoscedastic/'


#%% Read 
### ======================================
#read flatfile
df_flatfile = pd.read_csv(fn_fltfile)
#identify gm colums
i_flt_im = np.array([bool(re.match('^eas_f(.*)hz', c)) for c in df_flatfile.columns])
flt_freq = np.array([float(re.findall('eas_f(.*)hz', c)[0]) for c in df_flatfile.columns[i_flt_im]])
#identify unavailable gm
i_flt_gm_nan = np.isnan(df_flatfile.loc[:,i_flt_im].values)
#remove intensity measures
df_flatfile = df_flatfile.loc[:,~i_flt_im]

#read BA18 coefficients
df_ba18coeffs = pd.read_csv(fn_ba18coeffs)

#%% Processing Variables
### ======================================
#frequencies to keep
i_freq = np.arange(0, len(df_ba18coeffs.f), 10)
i_freq = np.append(i_freq, len(df_ba18coeffs.f)-1)
i_freq = np.unique(i_freq)

#reduce flatfile freq
df_ba18coeffs = df_ba18coeffs.loc[i_freq,:].reset_index(drop=True)

#identify unique events
_, eq_idx, eq_inv = np.unique(df_flatfile[['eqid']].values, axis=0, return_inverse=True, return_index=True)
#create earthquake ids for all records (1 to n_eq)
eq_id = eq_inv + 1
n_eq = len(eq_idx)

#identify unique stations
_, st_idx, st_inv = np.unique(df_flatfile[['stid']].values, axis=0, return_inverse=True, return_index=True)
#create stationfor all records (1 to n_eq)
st_id = st_inv + 1
n_st = len(st_idx)

#number of ground motions
n_gm = len(df_flatfile)

#ground motion parameters
# - - - - - - - - - - - - - -
#source parameters
mag  = df_flatfile.loc[:,'mag']
ztor = df_flatfile.loc[:,'ztor']
sof  = df_flatfile.loc[:,'sofid']
#path parameters
rrup = df_flatfile.loc[:,'rrup']
#site parameters
vs30 = df_flatfile.loc[:,'vs30']
z1p0 = np.array([calcBA19z1(vs30[j]) for j in range(n_gm)])

#regionalization
reg = df_flatfile.loc[:,'reg']


#%% Create Random Dataset
### ======================================
#colect aleatory std
aleat_sd = {}

#compute median ground moton
for j, f in enumerate(df_ba18coeffs.f):
    
    #linear coefficients
    c1   = df_ba18coeffs.loc[j,'c1']
    c2   = df_ba18coeffs.loc[j,'c2']
    c3   = df_ba18coeffs.loc[j,'c3']
    c4   = df_ba18coeffs.loc[j,'c4']
    c7   = df_ba18coeffs.loc[j,'c7']
    c8   = df_ba18coeffs.loc[j,'c8']
    c9   = df_ba18coeffs.loc[j,'c9']
    c10a = df_ba18coeffs.loc[j,'c10a']
    c10b = df_ba18coeffs.loc[j,'c10b']
    #non-linear coefficients
    c5   = df_ba18coeffs.loc[j,'c5']
    c6   = df_ba18coeffs.loc[j,'c6']
    cn   = df_ba18coeffs.loc[j,'cn']
    cmag = df_ba18coeffs.loc[j,'cM']
    chm  = df_ba18coeffs.loc[j,'chm']
    
    #aleatory variability
    s1    = df_ba18coeffs.loc[j,'s1']
    s2    = df_ba18coeffs.loc[j,'s2']
    s3    = df_ba18coeffs.loc[j,'s3']
    s4    = df_ba18coeffs.loc[j,'s4']
    s5    = df_ba18coeffs.loc[j,'s5']
    s6    = df_ba18coeffs.loc[j,'s6']
    #aleatory variability (mag break)
    s1mag = df_ba18coeffs.loc[j,'s1M']
    s2mag = df_ba18coeffs.loc[j,'s2M']
    s5mag = df_ba18coeffs.loc[j,'s5M']
    s6mag = df_ba18coeffs.loc[j,'s6M']
    
    #add regionaliztion in med scaling
    if flag_reg_med:
        reg_intrcp = np.array([reg_intrcp_scl[r] for r in reg ])
        reg_atten  = np.array([reg_atten_scl[r]  for r in reg])
        reg_vs30   = np.array([reg_vs30_scl[r]   for r in reg])
    else:
        reg_atten = np.ones(n_gm)
        reg_atten = np.ones(n_gm)
        reg_vs30  = np.ones(n_gm)
    
    #compute linear scaling terms
    intr, src, path, site, path_adj = gmm_eas_med(mag, ztor, sof, rrup, vs30, z1p0,
                                                  c5=c5, c6=c6,
                                                  cn=cn, cmag=cmag, chm=chm)
    
    #intercept
    f_intr = c1 * reg_intrcp * intr
    #source scaling
    f_src_lin = c2 * src[0] + c9 * src[2] + c10a * src[3] + c10b * src[4]  
    f_src     = c2 * src[0] + (c2-c3) * src[1] + c9 * src[2] + c10a * src[3] + c10b * src[4]  
    #path scaling
    f_path    = c4 * path[0] + c7 * reg_atten * path[1] + path_adj
    #site scaling
    f_site    = c8 * reg_vs30 * site[0]
    
    #compute median ground motion
    f_med = f_intr + f_src + f_path + f_site

    #compute aleatory variability
    if flag_hetero_sd:
        tau, tauP, phiS, phi = gmm_eas_sig(mag, s1, s2, s3, s4, s5, s6,
                                           s1mag, s2mag, s5mag, s6mag)

    else:
        tau  = np.full(n_gm, s1)
        tauP = np.full(n_gm, s3)
        phiS = np.full(n_gm, s4)
        phi  = np.full(n_gm, s5)

    #add regional adjustment
    if flag_reg_aleat:
        tau  *= 1.
        tauP *= 1.
        phiS *= np.array([reg_phiS_scl[r] for r in reg])
        phi  *= np.array([reg_phi_scl[r]  for r in reg])
        
    #store median scaling 
    df_flatfile.loc[:,'f_med_f%.9fhz'%f]     = f_med
    df_flatfile.loc[:,'f_src_f%.9fhz'%f]     = f_src
    df_flatfile.loc[:,'f_src_lin_f%.9fhz'%f] = f_src_lin
    df_flatfile.loc[:,'f_path_f%.9fhz'%f]    = f_path     
    df_flatfile.loc[:,'f_site_f%.9fhz'%f]    = f_site
    #store aleatory std
    df_flatfile.loc[:,'tau_f%.9fhz'%f]  = tau
    df_flatfile.loc[:,'tauP_f%.9fhz'%f] = tauP
    df_flatfile.loc[:,'phiS_f%.9fhz'%f] = phiS
    df_flatfile.loc[:,'phi_f%.9fhz'%f]  = phi

    #store aleatory std for rnd sampling
    aleat_sd['f%.2f'%f] = (tau[eq_idx], tauP[eq_idx], phiS[st_idx], phi)    
    

#sample variability
df_realiz_all = []
for l in range(n_realiz):
    print('Creating random realiztion: %3.i of %3.i'%(l+1, n_realiz))
    #create datataframe for rand realization
    df_realiz = df_flatfile.copy()

    #iterate over frequenices
    for j, f in enumerate(df_ba18coeffs.f):
        #aleatory std
        tau, tauP, phiS, phi = aleat_sd['f%.2f'%f]

        #sample aleatory variablity
        dB   = np.random.normal(0., tau,  n_eq)[eq_inv]
        dBP  = np.random.normal(0., tauP, n_eq)[eq_inv]
        dS   = np.random.normal(0., phiS, n_st)[st_inv]
        dWS  = np.random.normal(0., phi,  n_gm)
        #update unavailable values
        i_gm_nan_f = i_flt_gm_nan[:,np.argmin(np.abs(f-flt_freq))]
        dB[i_gm_nan_f]  = np.nan
        dBP[i_gm_nan_f] = np.nan
        dS[i_gm_nan_f]  = np.nan
        dWS[i_gm_nan_f] = np.nan
        
        #store aleatory samples
        df_realiz.loc[:,'dB_f%.9fhz'%f]  = dB
        df_realiz.loc[:,'dBP_f%.9fhz'%f] = dBP
        df_realiz.loc[:,'dS_f%.9fhz'%f]  = dS
        df_realiz.loc[:,'dWS_f%.9fhz'%f] = dWS
        #total aleatory samples
        df_realiz.loc[:,'dT_f%.9fhz'%f]  = dB + rrup * dBP + dS + dWS

        #compute response variables
        df_realiz.loc[:,'eas_f%.9fhz'%f]         = np.exp( df_realiz.loc[:,['f_med_f%.9fhz'%f,'dT_f%.9fhz'%f]].sum(axis=1) )  
        df_realiz.loc[:,'eas_lin_mag_f%.9fhz'%f] = np.exp( df_realiz.loc[:,['f_src_lin_f%.9fhz'%f,'f_path_f%.9fhz'%f,
                                                                            'f_site_f%.9fhz'%f,'dT_f%.9fhz'%f]].sum(axis=1) )
        df_realiz.loc[:,'eas_wo_mag_f%.9fhz'%f]  = np.exp( df_realiz.loc[:,['f_path_f%.9fhz'%f,'f_site_f%.9fhz'%f,
                                                                            'dT_f%.9fhz'%f]].sum(axis=1) )

        df_realiz_all.append(df_realiz)

# %% Save data
# ======================================
# create output directories
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

#dataframe with median scaling
fn_df_scl  = 'fltfile_med_scl'
df_flatfile.to_csv(dir_out+fn_df_scl+'.csv', index=False)

#dataframes with random realizations
for l in range(n_realiz):
    fn_df_rlz = 'fltfile_rlz%i'%(l+1)
    df_realiz_all[l].to_csv(dir_out+fn_df_rlz+'.csv', index=False)

