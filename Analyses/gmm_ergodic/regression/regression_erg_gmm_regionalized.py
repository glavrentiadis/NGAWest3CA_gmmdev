#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 07:43:06 2024

@author: glavrent
"""
#load libraries
#general
import os
import sys
import pathlib
#string libraries
import re
#arithmetic libraries
import numpy as np
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick
import arviz as az
#widgets
import ipywidgets
#stan library
import cmdstanpy
#user libraries
sys.path.insert(0,'../../python_lib')
from pylib_gmm import gmm_eas
from moving_mean import movingmean
import pylib_contour_plots as pycplt
#user functions
def get_posterior(param_samples, quantile=[0.05,0.16,0.84,0.95]):

    if param_samples.ndim == 1:
        param_mean    = np.mean(param_samples)
        param_median  = np.median(param_samples)
        param_sd      = np.std(param_samples)
        param_perc    = np.quantile(param_samples, q=quantile)
    elif param_samples.ndim == 2:
        param_mean    = np.mean(param_samples, axis=0)
        param_median  = np.median(param_samples, axis=0)
        param_sd      = np.std(param_samples, axis=0)
        param_perc    = np.quantile(param_samples, q=quantile, axis=0)

    #return posterior metrics
    return param_mean, param_median, param_sd, param_perc

#%% Define Variables
### ======================================
#response variable
freq = 5.011872
n_y  = 'eas_f%.9fhz'%freq

#region filter
# reg2process = ('ALL')
reg2process = ('PW','SEE','CH','JPN','TWN')
# reg2process = ('PW','AK','SEE','CH','JPN','TWN')

#max depth rupture
ztor_max = 20
#scaling path term
scl_atten = 0.01
scl_dBP   = 0.01

#max rupture distnace
rrup_max = 300

#flag MCMc sampling
flag_mcmc = True
#Optimizer
n_iter = 200000
#MCMC sampler
#chain iterations
# n_iter_warmup   = 20000
# n_iter_sampling = 10000
n_iter_warmup   = 250
n_iter_sampling = 250
# n_iter_warmup   = 50
# n_iter_sampling = 50
# n_iter_warmup   = 10
# n_iter_sampling = 10
#parameters
n_chains        = 6
adapt_delta     = 0.8
max_treedepth   = 14

#flag production
flag_production = True

#flag heteroscedasticity
flag_heteroscedastic = True

#
flag_upd_gs = True

#flag prior sensitivity
if flag_production: 
    flag_prior_sens = False
else:
    flag_prior_sens = True

#filename flatfile
if flag_production:
    fn_fltfile = '../../../Data/gmm_ergodic/dataset/fltfile_nga3_20240920_all.csv'
else:
    fn_fltfile = '../../../Data/gmm_ergodic/verification/dataset/homoscedastic/fltfile_rlz_1.csv'
    fn_fltfile = '../../../Data/gmm_ergodic/verification/dataset/heteroscedastic/fltfile_rlz_1.csv'

#filename coefficients prior
if flag_upd_gs:
    #filename updated gmm coefficients
    fn_upd_coeffs = '../../../Raw_files/coeff_20241021_mod.csv'
else:
    #filename BA18 coefficients
    fn_priorcoeffs = '../../../Raw_files/BA18coefs_mod.csv'

#stan model
fname_stan_model = '../../stan_lib/'
if flag_upd_gs:
    if not flag_heteroscedastic:
        fname_stan_model += 'regression_gmm_glob_regionalized_homoscedastic_upd_saturation_full_prior.stan'
    else:
        fname_stan_model += 'regression_gmm_glob_regionalized_heteroscedastic_upd_saturation_full_prior.stan'
else:
    if not flag_heteroscedastic:
        fname_stan_model += 'regression_gmm_glob_regionalized_homoscedastic_full_prior.stan'
    else:
        fname_stan_model += 'regression_gmm_glob_regionalized_heteroscedastic_full_prior.stan'

#define file output name
fname_out_main = 'eas_f%.4fhz'%freq

#output directory
if flag_production:
    dir_out = '../../../Data/gmm_ergodic/regression/'
else:
    dir_out = '../../../Data/gmm_ergodic/verification/regression/' 
    if flag_upd_gs:
        dir_out += 'updated_gs/'
    else:
        dir_out += 'original_gs/'
    if not flag_heteroscedastic:
        dir_out += 'homoscedastic_rlz1/'
    else:
        dir_out += 'heteroscedastic_rlz1/'

#figures output
dir_fig = dir_out + 'figures/'

#%% Read 
### ======================================
#read flatfile
df_flatfile = pd.read_csv(fn_fltfile)
#select regions to process
if not reg2process == 'ALL':
    df_flatfile = df_flatfile.loc[np.isin(df_flatfile.reg, reg2process),:].reset_index(drop=True)

#remove unavailable gms
df_flatfile = df_flatfile.loc[~np.isnan(df_flatfile[n_y]),:]
#apply rrup filter
df_flatfile = df_flatfile.loc[df_flatfile.rrup <= rrup_max,:]
#update indices
df_flatfile.reset_index(drop=False, inplace=True)

#read BA18 coefficients
df_coeffs_prior = pd.read_csv(fn_priorcoeffs)
#read updated coefficients
df_upd_coeffs  = pd.read_csv(fn_upd_coeffs)

#%% Processing Variables
### ======================================
#identify ground motion columns
i_gm = np.array([bool(re.match('(.*)hz', c)) for c in df_flatfile.columns])

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

#identify unique regions
_, reg_idx, req_inv = np.unique(df_flatfile[['regid']].values, return_inverse=True, return_index=True)
reg = df_flatfile.loc[reg_idx,'reg'].values
reg_id = req_inv + 1
n_reg = len(reg_idx)

#identify unique events within region
_, regeq_inv = np.unique(df_flatfile.loc[eq_idx,['regid']].values, return_inverse=True)
regeq_id = regeq_inv + 1

#identify unique station within region
_, regst_inv = np.unique(df_flatfile.loc[st_idx,['regid']].values, return_inverse=True)
regst_id = regst_inv + 1

#mean prior
c1_prior   = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c1'].values[0]   + flag_prior_sens * np.random.normal(0,.2)
c3_prior   = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c3'].values[0]   + flag_prior_sens * np.random.normal(0,.2)
c4_prior   = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c4'].values[0]   + flag_prior_sens * np.random.normal(0,.1)
c7_prior   = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c7'].values[0]   + flag_prior_sens * np.random.normal(0,.005)
c8_prior   = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c8'].values[0]   + flag_prior_sens * np.random.normal(0,.1)
c9_prior   = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c9'].values[0]   + flag_prior_sens * np.random.normal(0,.005)
c10a_prior = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c10a'].values[0] + flag_prior_sens * np.random.normal(0,.05)
c10b_prior = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c10b'].values[0] + flag_prior_sens * np.random.normal(0,.05)
#fixed coefficients
c2_fxd    = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c2'].values[0]
c5_fxd    = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c5'].values[0]
c6_fxd    = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'c6'].values[0]
cn_fxd    = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'cn'].values[0]
chm_fxd   = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'chm'].values[0]
cmag_fxd  = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'cM'].values[0]
#magnitude breaks
s1m = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'s1M'].values[0]
s2m = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'s2M'].values[0]
s5m = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'s5M'].values[0]
s6m = df_coeffs_prior.loc[df_coeffs_prior.f == freq,'s6M'].values[0]

#gmm parameters
mag  = df_flatfile.loc[:,'mag'].values
sof  = df_flatfile.loc[:,'sofid'].values
ztor = df_flatfile.loc[:,'ztor'].values
rrup = df_flatfile.loc[:,'rrup'].values
vs30 = df_flatfile.loc[:,'vs30'].values
z1p0 = np.zeros(n_gm)

#response variable
y = np.log(df_flatfile.loc[:,n_y].values)

#%% Regression
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

# regression input data
# ---   ---   ---   ---
#prepare regression data
stan_data = {'N':         n_gm,
             'NEQ':       n_eq,
             'NST':       n_st,
             'NREG':      n_reg,
             'eq':        eq_id,                  #earthquake id
             'st':        st_id,                  #station id
             'reg':       reg_id,
             'regeq':     regeq_id,
             'regst':     regst_id,
             'mag':       mag[eq_idx],
             'ztor':      ztor[eq_idx],
             'sof':       sof[eq_idx],
             'rrup':      rrup,
             'vs30':      vs30[st_idx],
             'Y':         y,
             #coefficient prior
             'c_1mu':     c1_prior,
             'c_3mu':     c3_prior,
             'c_4mu':     c4_prior,
             'c_7mu':     c7_prior,
             'c_8mu':     c8_prior,
             'c_9mu':     c9_prior,
             'c_10amu':   c10a_prior,
             'c_10bmu':   c10b_prior,
             #short distance saturation
             'c_2fxd':    c2_fxd,
             'c_5fxd':    c5_fxd,
             'c_6fxd':    c6_fxd,
             #mag break
             'c_nfxd':    cn_fxd,
             'c_hmfxd':   chm_fxd,
             'c_magfxd':  cmag_fxd,
             #maximum top of rupture
             'ztor_max':  ztor_max,
             #scaling 
             'scl_atten': scl_atten, #anelastic attenuation
             'scl_dBP':   scl_dBP,   #between event aleat
             #aleatory mag breaks
             's_1mag':    s1m,
             's_2mag':    s2m,
             's_5mag':    s5m,
             's_6mag':    s6m
            }

#write as json file
fname_stan_data = dir_out + 'reg_stan_erg_gmm_scl_data' + '.json'
try:
    cmdstanpy.utils.jsondump(fname_stan_data, stan_data)
except AttributeError:
    cmdstanpy.utils.write_stan_json(fname_stan_data, stan_data)

# run stan
# ---   ---   ---   ---
#compile stan model
stan_model = cmdstanpy.CmdStanModel(stan_file=fname_stan_model) 
stan_model.compile(force=True)
if flag_mcmc:
    #run full MCMC sampler
    stan_fit = stan_model.sample(data=fname_stan_data, chains=n_chains, 
                                 iter_warmup=n_iter_warmup, iter_sampling=n_iter_sampling,
                                 seed=1, refresh=10, max_treedepth=max_treedepth, adapt_delta=adapt_delta,
                                 show_progress=True,  show_console=True, output_dir=dir_out+'stan_fit/')
else:
    #run optimization
    stan_fit = stan_model.optimize(data=fname_stan_data,
                                   iter=n_iter,
                                   show_console=True, output_dir=dir_out+'stan_opt/' )

#delete json files
fname_dir = np.array( os.listdir(dir_out) )
#velocity filenames
fname_json = fname_dir[ [bool(re.search('\.json$',f_d)) for f_d in fname_dir] ]
for f_j in fname_json: os.remove(dir_out + f_j)

#%% Postprocessing
### ======================================
#metadata columns
c_gm_meta = df_flatfile.columns[~i_gm].values
c_eq_meta = ['eqid', 'eventid', 'reg', 'regid', 
             'eqlat', 'eqlon', 'eqx', 'eqy', 'eqz', 'eqcltlat', 'eqcltlon', 'eqcltx', 'eqclty', 'eqcltz', 'stlat', 
             'mag', 'ztor', 'sof', 'sofid', 'strike', 'dip', 'rake', 'eqclass']
c_st_meta = ['stid', 'staid', 'reg', 'regid', 'stlat', 'stlon', 'stx', 'sty', 
             'vs30', 'vs30class', 'z1.0', 'z2.5', 'z1.0flag', 'z2.5flag']
#initiaize flatfile for sumamry of coefficient
df_gmm_gm = df_flatfile.loc[:,c_gm_meta]
df_gmm_eq = df_flatfile.loc[eq_idx,c_eq_meta].reset_index(drop=True)
df_gmm_st = df_flatfile.loc[st_idx,c_st_meta].reset_index(drop=True)

if flag_mcmc:
    # full prior
    # - - - - - - - - - - - - - -
    #coefficients
    c1_full                                         = stan_fit.stan_variable('c_1')
    c2_full, c3_full, c9_full, c10a_full, c10b_full = [stan_fit.stan_variable(c) for c in ['c_2','c_3','c_9','c_10a','c_10b']]
    c4_full, c5_full, c6_full, c7_full              = [stan_fit.stan_variable(c) for c in ['c_4','c_5','c_6','c_7']]
    c8_full                                         = stan_fit.stan_variable('c_8')
    c11_full                                        = np.full(n_chains*n_iter_sampling, [0.])
    #regional coefficients
    c1r_full, c7r_full, c8r_full = [stan_fit.stan_variable(c) for c in ['c_1r','c_7r','c_8r']]
    #magnitude transition
    cn_full, chm_full, cmag_full = [stan_fit.stan_variable(s) for s in ['c_n','c_hm','c_mag']]
    #aleatory variability
    s1_full, s2_full, s3_full, s4_full, s5_full, s6_full = [stan_fit.stan_variable(s) 
                                                            for s in ['s_1','s_2','s_3','s_4','s_5','s_6']]
    #regional aleatory variability
    s3r_full, s4r_full, s5r_full, s6r_full = [stan_fit.stan_variable(s) 
                                              for s in ['s_3r','s_4r','s_5r','s_6r']]
    
    #summarize residuals
    #between-event
    deltaB_full  = stan_fit.stan_variable('deltaB')[:,eq_inv]
    #between-event-path
    deltaBP_full = stan_fit.stan_variable('deltaBP')[:,eq_inv]
    #between-site
    deltaS_full  = stan_fit.stan_variable('deltaS')[:,st_inv]
    #within-event-site
    deltaWS_full = stan_fit.stan_variable('deltaWS')
    #total
    deltaT_full  = deltaB_full + deltaBP_full + deltaS_full

    # prior statistics
    # - - - - - - - - - - - - - -
    #coefficients
    c1_mu,   c1_med,   c1_sd,   c1_q   = get_posterior(c1_full)
    c2_mu,   c2_med,   c2_sd,   c2_q   = get_posterior(c2_full)
    c3_mu,   c3_med,   c3_sd,   c3_q   = get_posterior(c3_full)
    c4_mu,   c4_med,   c4_sd,   c4_q   = get_posterior(c4_full)
    c5_mu,   c5_med,   c5_sd,   c5_q   = get_posterior(c5_full)
    c6_mu,   c6_med,   c6_sd,   c6_q   = get_posterior(c6_full)
    c7_mu,   c7_med,   c7_sd,   c7_q   = get_posterior(c7_full)
    c8_mu,   c8_med,   c8_sd,   c8_q   = get_posterior(c8_full)
    c9_mu,   c9_med,   c9_sd,   c9_q   = get_posterior(c9_full)
    c10a_mu, c10a_med, c10a_sd, c10a_q = get_posterior(c10a_full)
    c10b_mu, c10b_med, c10b_sd, c10b_q = get_posterior(c10b_full)
    c11_mu,  c11_med,  c11_sd,  c11_q  = get_posterior(c11_full)
    #regional coefficients
    c1r_mu,  c1r_med,  c1r_sd,  c1r_q  = get_posterior(c1r_full)
    c7r_mu,  c7r_med,  c7r_sd,  c7r_q  = get_posterior(c7r_full)
    c8r_mu,  c8r_med,  c8r_sd,  c8r_q  = get_posterior(c8r_full)
    #magnitude transition
    cn_mu,   cn_med,   cn_sd,   cn_q   = get_posterior(cn_full)
    chm_mu,  chm_med,  chm_sd,  chm_q  = get_posterior(chm_full)
    cmag_mu, cmag_med, cmag_sd, cmag_q = get_posterior(cmag_full)
    #aleatory variability
    s1_mu,   s1_med,   s1_sd,   s1_q   = get_posterior(s1_full)
    s2_mu,   s2_med,   s2_sd,   s2_q   = get_posterior(s2_full)
    s3_mu,   s3_med,   s3_sd,   s3_q   = get_posterior(s3_full)
    s4_mu,   s4_med,   s4_sd,   s4_q   = get_posterior(s4_full)
    s5_mu,   s5_med,   s5_sd,   s5_q   = get_posterior(s5_full)
    s6_mu,   s6_med,   s6_sd,   s6_q   = get_posterior(s6_full)
    #regional aleatory variability
    s3r_mu,  s3r_med,  s3r_sd,  s3r_q   = get_posterior(s3r_full)
    s4r_mu,  s4r_med,  s4r_sd,  s4r_q   = get_posterior(s4r_full)
    s5r_mu,  s5r_med,  s5r_sd,  s5r_q   = get_posterior(s5r_full)
    s6r_mu,  s6r_med,  s6r_sd,  s6r_q   = get_posterior(s6r_full)
    #summarize residuals
    deltaB_mu,  deltaB_med,  deltaB_sd,  deltaB_q  = get_posterior(deltaB_full)
    deltaBP_mu, deltaBP_med, deltaBP_sd, deltaBP_q = get_posterior(deltaBP_full)
    deltaS_mu,  deltaS_med,  deltaS_sd,  deltaS_q  = get_posterior(deltaS_full)
    deltaWS_mu, deltaWS_med, deltaWS_sd, deltaWS_q = get_posterior(deltaWS_full)
    deltaT_mu,  deltaT_med,  deltaT_sd,  deltaT_q  = get_posterior(deltaT_full)
else:
    # point estimates
    # - - - - - - - - - - - - - -
    #summarize gmm terms
    #coefficients
    c1_mu                                 = stan_fit.stan_variable('c_1')
    c2_mu, c3_mu, c9_mu, c10a_mu, c10b_mu = [stan_fit.stan_variable(c) for c in ['c_2','c_3','c_9','c_10a','c_10b']]
    c4_mu, c5_mu, c6_mu, c7_mu            = [stan_fit.stan_variable(c) for c in ['c_4','c_5','c_6','c_7']]
    c8_mu                                 = stan_fit.stan_variable('c_8')
    c11_mu                                = 0.
    #regional terms
    c1r_mu, c7r_mu, c8r_mu = [stan_fit.stan_variable(c) for c in ['c_1r','c_7r','c_8r']]
    #magnitude transition
    cn_mu, chm_mu, cmag_mu = [stan_fit.stan_variable(s) for s in ['c_n','c_hm','c_mag']]
    #aleatory variability
    s1_mu, s2_mu, s3_mu, s4_mu, s5_mu, s6_mu = [stan_fit.stan_variable(s) for s in ['s1','s2','s3','s4','s5','s6']]
    #regional aleatory variability
    s1r_mu, s2r_mu, s3r_mu, s4r_mu, s5r_mu, s6r_mu = [stan_fit.stan_variable(s) 
                                                      for s in ['s1r','s2r','s3r','s4r','s5r','s6r']]
    
    #summarize residuals
    #between-event
    deltaB_mu  = stan_fit.stan_variable('deltaB')[eq_inv]
    #between-event-path
    deltaBP_mu = stan_fit.stan_variable('deltaBP')[eq_inv]
    #between-site
    deltaS_mu  = stan_fit.stan_variable('deltaS')[st_inv]
    #within-event
    deltaWS_mu = stan_fit.stan_variable('deltaWS')
    #total
    deltaT_mu  = deltaB_mu + deltaBP_mu + deltaS_mu


# Summarize GMM parameters
# - - - - - - - - - - - 
#constant terms
df_gmm = pd.DataFrame({'c2':c2_mu,'c3':c3_mu,'c4':c4_mu,'c5':c5_mu,'c6':c6_mu,
                       'c10a':c10a_mu,'c10b':c10b_mu,'c11':c11_mu},
                      index=['global'])
#regional terms
df_gmm = pd.concat([df_gmm, pd.DataFrame({'c1':c1r_mu,'c7':c7r_mu,'c8':c8r_mu},             index=reg)])
#magnitude scaling
df_gmm = pd.concat([df_gmm, pd.DataFrame({'cn':cn_mu,'chm':chm_mu,'cmag':cmag_mu},          index=['global'])], axis=1)
#aleatory variability
df_gmm = pd.concat([df_gmm, pd.DataFrame({'s1':s1_mu,'s2':s2_mu},                           index=['global'])], axis=1)
df_gmm = pd.concat([df_gmm, pd.DataFrame({'s3':s3r_mu,'s4':s4r_mu,'s5':s5r_mu,'s6':s6r_mu}, index=reg)],        axis=1)
#re-organize columns
df_gmm = df_gmm.loc[:,['c1','c2','c3','c4','c5','c6','c7','c8','c10a','c10b','c11',
                       'cn','chm','cmag',
                       's1','s2','s3','s4','s5','s6']]

# GMM prediction
# - - - - - - - - - - -
#mean
f_mu, tau_mu, tauP_mu, phiS_mu, phi_mu = gmm_eas(mag, ztor, sof, rrup, vs30, z1p0, 
                                                 reg=df_flatfile['reg'].values,
                                                 regions=reg,
                                                 c1=c1r_mu, c2=c2_mu, c3=c3_mu, c4=c4_mu, 
                                                 c5=c5_mu, c6=c6_mu, c7=c7r_mu, c8=c8r_mu, c9=c9_mu,
                                                 c10a=c10a_mu, c10b=c10b_mu, 
                                                 c11=c11_mu, z1p0_vs30_brk=[],
                                                 cn=cn_mu, cmag=cmag_mu, chm=chm_mu, ztor_max=ztor_max,
                                                 s1=s1_mu, s2=s2_mu, s3=s3r_mu, s4=s4r_mu, s5=s5r_mu, s6=s6r_mu,
                                                 s1mag=s1m, s2mag=s2m, s5mag=s5m, s6mag=s6m)
#median
if flag_mcmc:
    f_med, tau_med, tauP_med, phiS_med, phi_med = gmm_eas(mag, ztor, sof, rrup, vs30, z1p0, 
                                                          reg=df_flatfile['reg'].values,
                                                          regions=reg,
                                                          c1=c1r_med, c2=c2_med, c3=c3_med, c4=c4_med, 
                                                          c5=c5_med, c6=c6_med, c7=c7r_med, c8=c8r_med, c9=c9_med,
                                                          c10a=c10a_med, c10b=c10b_med, 
                                                          c11=c11_med, z1p0_vs30_brk=[],
                                                          cn=cn_med, cmag=cmag_med, chm=chm_med, ztor_max=ztor_max,
                                                          s1=s1_med, s2=s2_med, s3=s3r_med, s4=s4r_med, s5=s5r_med, s6=s6r_med,
                                                          s1mag=s1m, s2mag=s2m, s5mag=s5m, s6mag=s6m)

#summarize event datafile
predict_summary_eq = np.vstack((tau_mu[eq_idx], tau_med[eq_idx], 
                                tauP_mu[eq_idx], tauP_med[eq_idx], 
                                deltaB_mu[eq_idx],  deltaB_med[eq_idx],  deltaB_sd[eq_idx],
                                deltaBP_mu[eq_idx], deltaBP_med[eq_idx], deltaBP_sd[eq_idx])).T
columns_names_eq = ['tau_mu','tau_med',
                    'tauP_mu','tauP_med',
                    'deltaB_mu','deltaB_med','deltaB_sd',
                    'deltaBP_mu','deltaBP_med','deltaBP_sd']
df_predict_summary_eq = pd.DataFrame(predict_summary_eq, columns=columns_names_eq, index=df_gmm_eq.index)
df_predict_summary_eq = pd.merge(df_gmm_eq, df_predict_summary_eq, how='right', left_index=True, right_index=True)

#summarize station datafile
predict_summary_st = np.vstack((phiS_mu[st_idx], phiS_med[st_idx], 
                                deltaS_mu[st_idx],  deltaS_med[st_idx],  deltaS_sd[st_idx])).T
columns_names_st = ['phiS_mu','phiS_med',
                    'deltaS_mu','deltaS_med','deltaS_sd']
df_predict_summary_st = pd.DataFrame(predict_summary_st, columns=columns_names_st, index=df_gmm_st.index)
df_predict_summary_st = pd.merge(df_gmm_st, df_predict_summary_st, how='right', left_index=True, right_index=True)

#summarize ground motion datafile
predict_summary_gm = np.vstack((y, f_mu, f_med, 
                                tau_mu, tau_med, 
                                tauP_mu, tauP_med, 
                                phiS_mu, phiS_med, 
                                phi_mu, phi_med,
                                deltaB_mu,  deltaB_med,  deltaB_sd,
                                deltaBP_mu, deltaBP_med, deltaBP_sd,
                                deltaS_mu,  deltaS_med,  deltaS_sd,
                                deltaWS_mu, deltaWS_med, deltaWS_sd,
                                deltaT_mu,  deltaT_med,  deltaT_sd)).T
columns_names_gm = ['y','f_mu','f_med',
                    'tau_mu','tau_med',
                    'tauP_mu','tauP_med',
                    'phiS_mu','phiS_med',
                    'phi_mu','phi_med',
                    'deltaB_mu','deltaB_med','deltaB_sd',
                    'deltaBP_mu','deltaBP_med','deltaBP_sd',
                    'deltaS_mu','deltaS_med','deltaS_sd',
                    'deltaWS_mu','deltaWS_med','deltaWS_sd',
                    'deltaT_mu','deltaT_med','deltaT_sd']
df_predict_summary_gm = pd.DataFrame(predict_summary_gm, columns=columns_names_gm, index=df_gmm_gm.index)
df_predict_summary_gm = pd.merge(df_gmm_gm, df_predict_summary_gm, how='right', left_index=True, right_index=True)

#save gmm coefficients
fname_df = fname_out_main + (fname_out_main + '_summary_coeff' + '.csv').replace(' ','_')
df_gmm.to_csv(dir_out + fname_df, index=True)

#save ground motion and random effects
fname_df = fname_out_main + (fname_out_main + '_summary_gm').replace(' ','_')
df_predict_summary_gm.to_csv(dir_out + fname_df + '.csv', index=False)
fname_df = fname_out_main + (fname_out_main + '_summary_eq').replace(' ','_')
df_predict_summary_eq.to_csv(dir_out + fname_df + '.csv', index=False)
fname_df = fname_out_main + (fname_out_main + '_summary_st').replace(' ','_')
df_predict_summary_st.to_csv(dir_out + fname_df + '.csv', index=False)

# Evaluate Residuals
# - - - - - - - - - - -
cmap = plt.get_cmap("tab10")


#magnitude bins
mag_bins = np.arange(3, 8.6, 0.5)
mag_dBbin, _, mag_dBmu, _, mag_dB16prc, mag_dB84prc = movingmean(df_predict_summary_eq.deltaB_mu,
                                                                 df_predict_summary_eq.mag, mag_bins)
mag_dBPbin, _, mag_dBPmu, _, mag_dBP16prc, mag_dBP84prc = movingmean(df_predict_summary_eq.deltaBP_mu,
                                                                     df_predict_summary_eq.mag, mag_bins)
mag_dWSbin, _, mag_dWSmu, _, mag_dWS16prc, mag_dWS84prc = movingmean(df_predict_summary_gm.deltaWS_mu,
                                                                     df_predict_summary_gm.mag, mag_bins)

#rupture distance bins
rrup_bins = np.logspace(np.log10(1), np.log10(400), 6)
rrup_dWSbin, _, rrup_dWSmu, _, rrup_dWS16prc, rrup_dWS84prc = movingmean(df_predict_summary_gm.deltaWS_mu,
                                                                         df_predict_summary_gm.rrup, rrup_bins)

#vs30 bins
vs30_bins = np.logspace(np.log10(100), np.log10(2000), 6)
vs30_dSbin, _, vs30_dSmu, _, vs30_dS16prc, vs30_dS84prc = movingmean(df_predict_summary_st.deltaS_mu,
                                                                     df_predict_summary_st.vs30, vs30_bins)
vs30_dWSbin, _, vs30_dWSmu, _, vs30_dWS16prc, vs30_dWS84prc = movingmean(df_predict_summary_gm.deltaWS_mu,
                                                                         df_predict_summary_gm.vs30, vs30_bins)




#between-event residuals
fname_fig = (fname_out_main + '_deltaB_scl').replace(' ','_')
fig, ax = plt.subplots(figsize = (20,10), nrows=1, ncols=1)
for j, rid in enumerate(np.unique(df_predict_summary_eq.regid)):
    #region points
    i_r = df_predict_summary_eq.regid == rid
    #region name
    rn = df_predict_summary_eq.loc[i_r,'reg'].unique()[0]
    #region plot handle
    hl = ax.plot(df_predict_summary_eq.loc[i_r,'mag'], df_predict_summary_eq.loc[i_r,'deltaB_mu'], 
                 'o', color=cmap(j), label=rn)
hl = ax.plot(mag_dBbin, mag_dBmu, 's', color='black', label='Mean')
hl = ax.errorbar(mag_dBbin, mag_dBmu, yerr=np.abs(np.vstack((mag_dB16prc,mag_dB84prc)) - mag_dBmu),
                  capsize=8, fmt='none', ecolor='black', linewidth=2,
                  label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel(r'Magnitude', fontsize=32)
ax.set_ylabel(r'$\delta B$', fontsize=32)
ax.legend(loc='lower right', fontsize=30, ncols=2)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.set_ylim([-3, 3])
ax.set_yticks([-3.,-1.5,0,1.5,3.])
fig.tight_layout()
fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')

#between-event-path residuals
fname_fig = (fname_out_main + '_deltaBP_scl').replace(' ','_')
fig, ax = plt.subplots(figsize = (20,10), nrows=1, ncols=1)
for j, rid in enumerate(np.unique(df_predict_summary_eq.regid)):
    #region points
    i_r = df_predict_summary_eq.regid == rid
    #region name
    rn = df_predict_summary_eq.loc[i_r,'reg'].unique()[0]
    #region plot handle
    hl = ax.plot(df_predict_summary_eq.loc[i_r,'mag'], 1/scl_dBP*df_predict_summary_eq.loc[i_r,'deltaBP_mu'], 
                 'o', color=cmap(j), label=rn)
hl = ax.plot(mag_dBPbin, 1/scl_dBP*mag_dBPmu, 's', color='black', label='Mean')
hl = ax.errorbar(mag_dBPbin, 1/scl_dBP*mag_dBPmu, 
                 yerr=1/scl_dBP*np.abs(np.vstack((mag_dBP16prc,mag_dBP84prc)) - mag_dBPmu),
                 capsize=8, fmt='none', ecolor='black', linewidth=2,
                 label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel(r'Magnitude', fontsize=32)
ax.set_ylabel(r'$\delta BP$ (scaled)', fontsize=32)
ax.legend(loc='lower right', fontsize=30, ncols=2)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.set_ylim([-3, 3])
ax.set_yticks([-3.,-1.5,0,1.5,3.])
fig.tight_layout()
fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')

#between-site residuals
fname_fig = (fname_out_main + '_deltaS_scl').replace(' ','_')
fig, ax = plt.subplots(figsize = (20,10), nrows=1, ncols=1)
for j, rid in enumerate(np.unique(df_predict_summary_st.regid)):
    #region points
    i_r = df_predict_summary_st.regid == rid
    #region name
    rn = df_predict_summary_st.loc[i_r,'reg'].unique()[0]
    #region plot handle
    hl = ax.semilogx(df_predict_summary_st.loc[i_r,'vs30'], df_predict_summary_st.loc[i_r,'deltaS_mu'], 
                     'o', color=cmap(j), label=rn)
hl = ax.plot(vs30_dSbin, vs30_dSmu, 's', color='black', label='Mean')
hl = ax.errorbar(vs30_dSbin, vs30_dSmu, yerr=np.abs(np.vstack((vs30_dS16prc,vs30_dS84prc)) - vs30_dSmu),
                 capsize=8, fmt='none', ecolor='black', linewidth=2,
                 label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel(r'$V_{S30}$',   fontsize=32)
ax.set_ylabel(r'$\delta S$', fontsize=32)
ax.legend(loc='lower right', fontsize=30, ncols=2)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.set_xlim([100, 2500])
ax.set_ylim([-3, 3])
ax.set_yticks([-3.,-1.5,0,1.5,3.])
fig.tight_layout()
fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')

#within-event-site
fname_fig = (fname_out_main + '_deltaWS_scl').replace(' ','_')
#within-event-site (mag scaling)
fig, ax = plt.subplots(figsize = (20,3*10), nrows=3, ncols=1)
for j, rid in enumerate(np.unique(df_predict_summary_gm.regid)):
    #region points
    i_r = df_predict_summary_gm.regid == rid
    #region name
    rn = df_predict_summary_gm.loc[i_r,'reg'].unique()[0]
    #region plot handle
    hl = ax[0].plot(df_predict_summary_gm.loc[i_r,'mag'], df_predict_summary_gm.loc[i_r,'deltaWS_mu'], 
                    'o', color=cmap(j), label=rn)
hl = ax[0].plot(mag_dWSbin, mag_dWSmu, 's', color='black', label='Mean')
hl = ax[0].errorbar(mag_dWSbin, mag_dWSmu, yerr=np.abs(np.vstack((mag_dWS16prc,mag_dWS84prc)) - mag_dWSmu),
                    capsize=8, fmt='none', ecolor='black', linewidth=2,
                    label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax[0].set_xlabel(r'Magnitude',   fontsize=32)
ax[0].set_ylabel(r'$\delta WS$', fontsize=32)
ax[0].legend(loc='lower right', fontsize=30, ncols=2)
ax[0].grid(which='both')
ax[0].tick_params(axis='x', labelsize=30)
ax[0].tick_params(axis='y', labelsize=30)
# ax[0].set_ylim([-3, 3])
# ax[0].set_yticks([-3.,-1.5,0,1.5,3.])
#within-event-site (rrup scaling)
for j, rid in enumerate(np.unique(df_predict_summary_gm.regid)):
    #region points
    i_r = df_predict_summary_gm.regid == rid
    #region name
    rn = df_predict_summary_gm.loc[i_r,'reg'].unique()[0]
    #region plot handle
    hl = ax[1].semilogx(df_predict_summary_gm.loc[i_r,'rrup'], df_predict_summary_gm.loc[i_r,'deltaWS_mu'], 
                     'o', color=cmap(j), label=rn)
hl = ax[1].semilogx(rrup_dWSbin, rrup_dWSmu, 's', color='black', label='Mean')
hl = ax[1].errorbar(rrup_dWSbin, rrup_dWSmu, yerr=np.abs(np.vstack((rrup_dWS16prc,rrup_dWS84prc)) - rrup_dWSmu),
                    capsize=8, fmt='none', ecolor='black', linewidth=2,
                    label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax[1].set_xlabel(r'$R_{rup}$ km', fontsize=32)
ax[1].set_ylabel(r'$\delta WS$',  fontsize=32)
#ax[1].legend(loc='lower right', fontsize=30)
ax[1].grid(which='both')
ax[1].tick_params(axis='x', labelsize=30)
ax[1].tick_params(axis='y', labelsize=30)
# ax[1].set_ylim([-3, 3])
# ax[1].set_yticks([-3.,-1.5,0,1.5,3.])
#within-event-site (vs30 scaling)
for j, rid in enumerate(np.unique(df_predict_summary_gm.regid)):
    #region points
    i_r = df_predict_summary_gm.regid == rid
    #region name
    rn = df_predict_summary_gm.loc[i_r,'reg'].unique()[0]
    #region plot handle
    hl = ax[2].semilogx(df_predict_summary_gm.loc[i_r,'vs30'], df_predict_summary_gm.loc[i_r,'deltaWS_mu'], 
                     'o', color=cmap(j), label=rn)
hl = ax[2].plot(vs30_dWSbin, vs30_dWSmu, 's', color='black', label='Mean')
hl = ax[2].errorbar(vs30_dWSbin, vs30_dWSmu, yerr=np.abs(np.vstack((vs30_dWS16prc,vs30_dWS84prc)) - vs30_dWSmu),
                    capsize=8, fmt='none', ecolor='black', linewidth=2,
                    label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax[2].set_xlabel(r'$V_{S30}$',   fontsize=32)
ax[2].set_ylabel(r'$\delta WS$', fontsize=32)
#ax[2].legend(loc='lower right', fontsize=30)
ax[2].grid(which='both')
ax[2].tick_params(axis='x', labelsize=30)
ax[2].tick_params(axis='y', labelsize=30)
ax[2].set_xlim([100, 2500])
# ax[2].set_ylim([-3, 3])
# ax[2].set_yticks([-3.,-1.5,0,1.5,3.])
fig.tight_layout()
fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')

#between-event residuals (spatial variability)
for j, rid in enumerate(np.unique(df_predict_summary_eq.regid)):
    #region points
    i_r = df_predict_summary_eq.regid == rid
    #region name
    rn = df_predict_summary_eq.loc[i_r,'reg'].unique()[0]
    #figure name
    fname_fig = (fname_out_main + '_deltaB_reg_' + rn).replace(' ','_')
    #deltaB
    data2plot = df_predict_summary_eq.loc[i_r,['eqlat','eqlon','deltaB_mu']].values
    #plot figure
    fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=-3.0,  cmax=3.0, flag_grid=False, 
                                                          title=None, cbar_label='', log_cbar = False, 
                                                          frmt_clb = '%.2f', alpha_v = 0.7, cmap='seismic', 
                                                          marker_size=70.)
    #edit figure properties
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label(r'Median: $\delta B$', size=30)
    #grid lines
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 28}
    gl.ylabel_style = {'size': 28}
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png', bbox_inches='tight')

#between-event-path residuals (spatial variability)
for j, rid in enumerate(np.unique(df_predict_summary_eq.regid)):
    #region points
    i_r = df_predict_summary_eq.regid == rid
    #region name
    rn = df_predict_summary_eq.loc[i_r,'reg'].unique()[0]
    #figure name
    fname_fig = (fname_out_main + '_deltaBP_reg_' + rn).replace(' ','_')
    #deltaB
    data2plot = df_predict_summary_eq.loc[i_r,['eqlat','eqlon','deltaBP_mu']].values
    data2plot[:,2] /= scl_dBP
    #plot figure
    fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=-3.0, cmax=3.0, flag_grid=False, 
                                                          title=None, cbar_label='', log_cbar = False, 
                                                          frmt_clb = '%.2f', alpha_v = 0.7, cmap='seismic', 
                                                          marker_size=70.)
    #edit figure properties
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label(r'Median: $\delta BP$ (scaled)', size=30)
    #grid lines
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 28}
    gl.ylabel_style = {'size': 28}
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png', bbox_inches='tight')

#between-site residuals (spatial variability)
for j, rid in enumerate(np.unique(df_predict_summary_st.regid)):
    #region points
    i_r = df_predict_summary_st.regid == rid
    #region name
    rn = df_predict_summary_st.loc[i_r,'reg'].unique()[0]
    #figure name
    fname_fig = (fname_out_main + '_deltaS_reg_' + rn).replace(' ','_')
    #deltaB
    data2plot = df_predict_summary_st.loc[i_r,['stlat','stlon','deltaS_mu']].values
    #plot figure
    fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=-3.0, cmax=3.0, flag_grid=False, 
                                                          title=None, cbar_label='', log_cbar = False, 
                                                          frmt_clb = '%.2f', alpha_v = 0.7, cmap='seismic', 
                                                          marker_size=70.)
    #edit figure properties
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label(r'Median: $\delta S$', size=30)
    #grid lines
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 28}
    gl.ylabel_style = {'size': 28}
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png', bbox_inches='tight')
    
#within-event-site residuals (spatial variability)
for j, rid in enumerate(np.unique(df_predict_summary_gm.regid)):
    #region points
    i_r = df_predict_summary_gm.regid == rid
    #region name
    rn = df_predict_summary_gm.loc[i_r,'reg'].unique()[0]
    #figure name
    fname_fig = (fname_out_main + '_deltaWS_reg_' + rn).replace(' ','_')
    #deltaB
    data2plot = df_predict_summary_gm.loc[i_r,['stlat','stlon','deltaWS_mu']].values
    #plot figure
    fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=-3.0, cmax=3.0, flag_grid=False, 
                                                          title=None, cbar_label='', log_cbar = False, 
                                                          frmt_clb = '%.2f', alpha_v = 0.7, cmap='seismic', 
                                                          marker_size=70.)
    #edit figure properties
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label(r'Median: $\delta WS$', size=30)
    #grid lines
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 28}
    gl.ylabel_style = {'size': 28}
    #save figure
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png', bbox_inches='tight')
    
# Summary regression
# ---------------------------
#save summary statistics
stan_summary_fname = dir_out  + fname_out_main + '_stan_summary' + '.txt'
with open(stan_summary_fname, 'w') as f:
    print(stan_fit, file=f)

#create and save trace plots
dir_fig = dir_out  + 'summary_figs/'
#create figures directory if doesn't exit
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#create stan trace plots
stan_az_fit = az.from_cmdstanpy(stan_fit)
for c_vn, c_n in zip(['c_1','c_2','c_3','c_4','c_7','c_8','c_9','c_10a','c_10b'], 
                     ['$c_1$','$c_2$','$c_3$','$c_4$','$c_7$','$c_8$','$c_9$','$c_10a$','$c_10b$',]):
    #create trace plot with arviz
    ax = az.plot_trace(stan_az_fit,  var_names=c_vn, figsize=(10,5) ).ravel()
    ax[0].yaxis.set_major_locator(plt_autotick())
    ax[0].set_xlabel('sample value')
    ax[0].set_ylabel('frequency')
    ax[0].set_title('')
    ax[0].grid(axis='both')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('sample value')
    ax[1].grid(axis='both')
    ax[1].set_title('')
    fig = ax[0].figure
    fig.suptitle(c_n)
    fig.savefig(dir_fig + fname_out_main + '_stan_traceplot_' + c_vn + '_arviz' + '.png')

