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
#user libraries
sys.path.insert(0,'../../python_lib')
from pylib_gmm import gmm_eas, gmm_eas_upd
from pylib_gmm import scalingHW as scaling_hw 
from pylib_gmm import calcBA19z1 as calc_ba19_z1

#suppress warning
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#%% Define Variables
### ======================================
#heteroscedasticity option
flag_hetero_sd = True
#hanging wall
flag_hw = True

#regionaliztion option
#median scaling
flag_reg_intrcp = True
flag_reg_smag   = True
flag_reg_atten  = True
flag_reg_vs30   = True
#aleatory varibility
flag_reg_tau0   = True
flag_reg_tauP   = True
flag_reg_phiS   = True
flag_reg_phi0   = True
#updated short-distance geometrical spreading
flag_upd_satur  = True

#correlation lengths
phiS_ell = 10

#number of random realizations
n_realiz = 10

#regionaliztion of intercpet
reg_intrcp_scl = {'PW': 1.00,
                  'AK': 1.05,
                  'SEE':1.10, 
                  'CH': 0.95, 
                  'JPN':0.80,  
                  'NZ': 1.15,  
                  'TWN':0.90}

#regionaliztion of small magnitude scaling
reg_smag_scl =   {'PW': 1.00,
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
reg_tau0_scl =  {'PW': 1.00,
                 'AK': 1.05,
                 'SEE':1.10, 
                 'CH': 0.95, 
                 'JPN':0.80,  
                 'NZ': 1.15,  
                 'TWN':0.90}

reg_tauP_scl =  {'PW': 1.00,
                 'AK': 1.05,
                 'SEE':1.10, 
                 'CH': 0.95, 
                 'JPN':0.80,  
                 'NZ': 1.15,  
                 'TWN':0.90}

reg_phi0_scl = {'PW': 1.00,
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

#ensure regionalization consistency
assert(reg_smag_scl.keys()  == reg_intrcp_scl.keys()),"Error. Inconsistent keys."
assert(reg_atten_scl.keys() == reg_intrcp_scl.keys()),"Error. Inconsistent keys."
assert(reg_vs30_scl.keys()  == reg_intrcp_scl.keys()),"Error. Inconsistent keys."
assert(reg_tau0_scl.keys()  == reg_intrcp_scl.keys()),"Error. Inconsistent keys."
assert(reg_tauP_scl.keys()  == reg_intrcp_scl.keys()),"Error. Inconsistent keys."
assert(reg_phi0_scl.keys()  == reg_intrcp_scl.keys()),"Error. Inconsistent keys."
assert(reg_phiS_scl.keys()  == reg_intrcp_scl.keys()),"Error. Inconsistent keys."

#flatfiel filename
fn_fltfile = '../../../Data/gmm_ergodic/dataset/fltfile_nga3_20240704_rev1.csv'
fn_fltfile = '../../../Data/gmm_ergodic/dataset/fltfile_nga3_20240920_all.csv'
fn_fltfile = '../../../Data/gmm_ergodic/dataset/fltfile_nga3_20240920_censor.csv'

#previous GMM coefficients
if not flag_upd_satur:
    fn_coeffs = '../../../Raw_files/model_coeffs/BA18coefs_mod.csv'
else:
    fn_coeffs = '../../../Raw_files/model_coeffs/coeff_20241021_mod2.csv'

#output directory
dir_out = '../../../Data/gmm_ergodic/verification/dataset/'
dir_out += 'updated_saturation/' if flag_upd_satur else 'original_saturation/'
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

#read coefficients
df_coeffs  = pd.read_csv(fn_coeffs)
#reduce
if not flag_upd_satur:
    #frequencies to keep
    i_freq = np.arange(0, len(df_coeffs.f), 10)
    i_freq = np.append(i_freq, len(df_coeffs.f)-1)
    i_freq = np.unique(i_freq)
    #reduce flatfile freq
    df_coeffs = df_coeffs.loc[i_freq,:].reset_index(drop=True)


#%% Processing Variables
### ======================================
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
mag   = df_flatfile.loc[:,'mag'].values
ztor  = df_flatfile.loc[:,'ztor'].values
width = df_flatfile.loc[:,'width'].values
dip   = df_flatfile.loc[:,'dip'].values
sof   = df_flatfile.loc[:,'sofid'].values
#path parameters
rrup  = df_flatfile.loc[:,'rrup'].values
rjb   = df_flatfile.loc[:,'rjb'].values
rx    = df_flatfile.loc[:,'rx'].values
ry    = df_flatfile.loc[:,'ry'].values
ry0   = df_flatfile.loc[:,'ry0'].values
#site parameters
vs30  = df_flatfile.loc[:,'vs30'].values
z1p0  = np.array([calc_ba19_z1(vs30[j]) for j in range(n_gm)])

#regionalization
reg = df_flatfile.loc[:,'reg'].values


#%% Create Random Dataset
### ======================================
#colect aleatory std
aleat_sd = {}

#compute median ground moton
for j, f in enumerate(df_coeffs.f):
    
    #linear coefficients
    c1   = df_coeffs.loc[j,'c1']
    c2   = df_coeffs.loc[j,'c2']
    c3   = df_coeffs.loc[j,'c3']
    c4   = df_coeffs.loc[j,'c4']
    c7   = df_coeffs.loc[j,'c7']
    c8   = df_coeffs.loc[j,'c8']
    c9   = df_coeffs.loc[j,'c9']
    c10a = df_coeffs.loc[j,'c10a']
    c10b = df_coeffs.loc[j,'c10b']
    c13  = df_coeffs.loc[j,'c13']
    #non-linear coefficients
    c5   = df_coeffs.loc[j,'c5']
    c6   = df_coeffs.loc[j,'c6']
    chm  = df_coeffs.loc[j,'chm']
    cn   = df_coeffs.loc[j,'cn']
    cmag = df_coeffs.loc[j,'cM']
    
    #aleatory variability
    s1    = df_coeffs.loc[j,'s1']
    s2    = df_coeffs.loc[j,'s2']
    s3    = df_coeffs.loc[j,'s3']
    s4    = df_coeffs.loc[j,'s4']
    s5    = df_coeffs.loc[j,'s5']
    s6    = df_coeffs.loc[j,'s6']
    #aleatory variability (mag break)
    s1mag = df_coeffs.loc[j,'s1M']
    s2mag = df_coeffs.loc[j,'s2M']
    s5mag = df_coeffs.loc[j,'s5M']
    s6mag = df_coeffs.loc[j,'s6M']
    
    #regionalized coefficients
    c1r = c1 * (np.array([reg_intrcp_scl[r] for r in reg ]) if flag_reg_intrcp else np.ones(len(reg)) )
    c3r = c3 * (np.array([reg_smag_scl[r] for r in reg ])   if flag_reg_smag   else np.ones(len(reg)) )
    c7r = c7 * (np.array([reg_atten_scl[r] for r in reg ])  if flag_reg_atten  else np.ones(len(reg)) )
    c8r = c8 * (np.array([reg_vs30_scl[r] for r in reg ])   if flag_reg_vs30   else np.ones(len(reg)) )
    
    #regionalized aleatory terms
    s1r = s1 * (np.array([reg_tau0_scl[r] for r in reg ]) if reg_tau0_scl else np.ones(len(reg)) )
    s3r = s3 * (np.array([reg_tauP_scl[r] for r in reg ]) if reg_tauP_scl else np.ones(len(reg)) )
    s4r = s4 * (np.array([reg_phiS_scl[r] for r in reg ]) if reg_phiS_scl else np.ones(len(reg)) )
    s5r = s5 * (np.array([reg_phi0_scl[r] for r in reg ]) if reg_phi0_scl else np.ones(len(reg)) )
    s6r = s6 * (np.array([reg_phi0_scl[r] for r in reg ]) if reg_phi0_scl else np.ones(len(reg)) )
    
    #remove heteroscedasticity
    if not flag_hetero_sd:
        s1r = s2
        s5r = s6r  
    
    #compute ground motion
    if not flag_upd_satur:
        f_med, f_src, f_path, f_site, tau0, tauP, phiS, phi0 = gmm_eas(mag, ztor, sof, rrup, vs30, z1p0, 
                                                                       reg=reg,
                                                                       regions=list(reg_intrcp_scl.keys()),
                                                                       c1=c1r, c2=c2, c3=c3r, c4=c4, 
                                                                       c5=c5, c6=c6, c7=c7r, c8=c8r, c9=c9,
                                                                       c10a=c10a, c10b=c10b, 
                                                                       c11=0., z1p0_vs30_brk=[],
                                                                       cn=cn, cmag=cmag, chm=chm, ztor_max=20.,
                                                                       s1=s1r, s2=s2, s3=s3r, s4=s4r, s5=s5r, s6=s6r,
                                                                       s1mag=s1mag, s2mag=s2mag, s5mag=s5mag, s6mag=s6mag)
    else:
        f_med, f_src, f_path, f_site, tau0, tauP, phiS, phi0 = gmm_eas_upd(mag, ztor, sof, rrup, vs30, z1p0, 
                                                                           reg=reg,
                                                                           regions=list(reg_intrcp_scl.keys()),
                                                                           c1=c1r, c2=c2, c3=c3r, c4=c4, 
                                                                           c5=c5, c6=c6, c7=c7r, c8=c8r, c9=c9,
                                                                           c10a=c10a, c10b=c10b, 
                                                                           c11=0., z1p0_vs30_brk=[],
                                                                           cn=cn, cmag=cmag, chm=chm, ztor_max=20.,
                                                                           s1=s1r, s2=s2, s3=s3r, s4=s4r, s5=s5r, s6=s6r,
                                                                           s1mag=s1mag, s2mag=s2mag, s5mag=s5mag, s6mag=s6mag)
    
    #hanging all scaing
    if flag_hw:
        f_hw = scaling_hw(mag, width, dip, ztor, rjb, rrup, rx, ry)
    else:
        f_hw = np.zeros(n_gm)
    f_med += f_hw
    
    #store median scaling 
    df_flatfile.loc[:,'f_med_f%.9fhz'%f]     = f_med
    df_flatfile.loc[:,'f_src_f%.9fhz'%f]     = f_src
    df_flatfile.loc[:,'f_path_f%.9fhz'%f]    = f_path     
    df_flatfile.loc[:,'f_site_f%.9fhz'%f]    = f_site
    df_flatfile.loc[:,'f_hw_f%.9fhz'%f]      = f_hw
    #store aleatory std
    df_flatfile.loc[:,'tau0_f%.9fhz'%f] = tau0
    df_flatfile.loc[:,'tauP_f%.9fhz'%f] = tauP
    df_flatfile.loc[:,'phiS_f%.9fhz'%f] = phiS
    df_flatfile.loc[:,'phi0_f%.9fhz'%f] = phi0

    #store aleatory std for rnd sampling
    aleat_sd['f%.9fhz'%f] = (tau0[eq_idx], tauP[eq_idx], phiS[st_idx], phi0)    
    

#sample variability
df_realiz_all = []
for l in range(n_realiz):
    print('Creating random realiztion: %3.i of %3.i'%(l+1, n_realiz))
    #create datataframe for rand realization
    df_realiz = df_flatfile.copy()

    #iterate over frequenices
    for j, f in enumerate(df_coeffs.f):
        #aleatory std
        tau0, tauP, phiS, phi0 = aleat_sd['f%.9fhz'%f]

        #sample aleatory variablity
        dB   = np.random.normal(0., tau0, n_eq)[eq_inv]
        dBP  = np.random.normal(0., tauP, n_eq)[eq_inv]
        dS   = np.random.normal(0., phiS, n_st)[st_inv]
        dWS  = np.random.normal(0., phi0, n_gm)
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
