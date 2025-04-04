#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:57:06 2024

@author: glavrent
"""

#load libraries
#general
import os
import sys
import shutil
import pathlib
import platform
import datetime
#string libraries
import re
#arithmetic libraries
import numpy as np
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
import arviz as az
#widgets
import ipywidgets
#stan library
import cmdstanpy
#user libraries
sys.path.insert(0,'../../python_lib')
from pylib_gmm_plots import figures_residuals
from pylib_gmm_plots import figures_residuals_adjust
#user functions
def adjust_residuals(df_totres, stan_model,
                     s1, s2, s3, s4, s5, s6, s1mag, s2mag, s5mag, s6mag,
                     scl_dBP=100., rrup_offset_dBP=50,
                     cname_dT='dT', 
                     cname_dB='dB', cname_dBP='dBP', cname_dS='dS',
                     n_iter=10000):
    
    # residual calculation
    # ---   ---   ---   ---   ---
    #region name
    reg_n = np.unique(df_totres.reg)
    reg_n = reg_n[0] if len(reg_n) == 1 else 'ALL'
 
    #number of data points
    n_data = len(df_totres)

    #earthquake data
    data_eq_all = df_totres[['eqid','mag']].values
    _, eq_idx, eq_inv, eq_cnt = np.unique(df_totres[['eqid']], axis=0, return_inverse=True, return_index=True, return_counts=True)
    data_eq = data_eq_all[eq_idx,:]
    #create earthquake ids for all records (1 to n_eq)
    eq_id = eq_inv + 1
    n_eq = len(data_eq)

    #station data
    data_sta_all = df_totres[['stid','vs30']].values
    _, sta_idx, sta_inv, sta_cnt = np.unique( df_totres[['stid']].values, axis = 0, return_inverse=True, return_index=True, return_counts=True)
    data_sta = data_sta_all[sta_idx,:]
    #create station indices for all records (1 to n_sta)
    sta_id = sta_inv + 1
    n_sta = len(data_sta)

    #magnitude
    mag  = data_eq[:,1]
    #rupture distance
    rrup = df_totres.loc[:,'rrup'].values

    #total residuals
    deltaT = df_totres[cname_dT].to_numpy().copy()
    
    # residual calculation
    # ---   ---   ---   ---   ---
    #prepare regression data
    stan_data = {'N':                n_data,
                 'NEQ':              n_eq,
                 'NSTA':             n_sta,
                 'eq':               eq_id,   #earthquake id
                 'sta':              sta_id,  #station id
                 #gm parameters
                 'mag':              mag,
                 'rrup':             rrup,
                 #total resisuals
                 'Y':                deltaT,
                 #aleatory scaling
                 's_1':              s1,
                 's_2':              s2,
                 's_3':              s3,
                 's_4':              s4,
                 's_5':              s5,
                 's_6':              s6,
                 #aleatory mag breaks
                 's_1mag':           s1mag,
                 's_2mag':           s2mag,
                 's_5mag':           s5mag,
                 's_6mag':           s6mag,
                 #scaling 
                 'scl_dBP':          scl_dBP,   #between event aleat
                 'rrup_offset_dBP': rrup_offset_dBP,
                 #penality factor
                 'lambda':           lambda_std
                }
    
    #initial parameter values
    stan_inits = {'deltaB':      df_totres[cname_dB].to_numpy()[eq_idx].copy(),
                  'deltaS':      df_totres[cname_dS].to_numpy()[sta_idx].copy(),
                  'deltaBP_scl': 1/scl_dBP * df_totres[cname_dBP].to_numpy()[eq_idx].copy(),
                  }
  
    #write as json file
    fname_stan_data = dir_out + 'reg_stan_res_partition_f%.4fhz_reg_%s'%(freq, reg_n) + '.json'
    try:
        cmdstanpy.utils.jsondump(fname_stan_data, stan_data)
    except AttributeError:
        cmdstanpy.utils.write_stan_json(fname_stan_data, stan_data)
        
    #run optimization
    stan_fit = stan_model.optimize(data=fname_stan_data,
                                   inits=stan_inits,
                                   iter=n_iter,
                                   show_console=True, output_dir=dir_out+'stan_opt/' )
    
    # extract posterior samples
    # ---   ---   ---   ---   ---
    #identity ground motion columns
    i_gm = np.any(np.array([[bool(re.match(pattern, c)) for c in df_totres.columns.values]
                            for pattern in ('delta.*','tau.*','phi.*')]), axis=0)
    #ground motion information
    c_gm_meta = df_totres.columns[~i_gm].values
    #summarize ground motion information
    df_gminfo = df_totres[c_gm_meta]
    #convert region, event and station ids to integers
    df_gminfo.loc[:,['eqid','eventid','regid','stid','stationid']] =\
        df_gminfo[['eqid','eventid','regid','stid','stationid']].astype(int)
    
    #collect standard deviations
    #between-event
    tau0  = stan_fit.stan_variable('tau0')[eq_inv]
    #between-event-path
    tauP = stan_fit.stan_variable('tauP') * np.ones(n_data)
    #between-site
    phiS = stan_fit.stan_variable('phiS') * np.ones(n_data)
    #within-event-site
    phi0 = stan_fit.stan_variable('phi0')[eq_inv]

    #collect random effects
    #between-event
    deltaB  = stan_fit.stan_variable('deltaB')[eq_inv]
    #between-event-path
    deltaBP = stan_fit.stan_variable('deltaBP')[eq_inv]
    #between-site
    deltaS  = stan_fit.stan_variable('deltaS')[sta_inv]
    #within-event-site
    deltaWS = stan_fit.stan_variable('deltaWS')
    
    #summarize random effects
    rndeff_summary = np.vstack((tau0, tauP, phiS, phi0, 
                                deltaB, deltaBP,
                                deltaS, deltaWS,
                                deltaT)).T
    columns_names = ['tau0', 'tauP', 'phiS', 'phi0',
                     'deltaB', 'deltaBP', 'deltaS', 'deltaWS', 'deltaT']
    df_rndeff_summary = pd.DataFrame(rndeff_summary, columns = columns_names, index=df_gminfo.index)
    #create dataframe with random effect summary
    df_rndeff_summary = pd.merge(df_gminfo, df_rndeff_summary, how='right', left_index=True, right_index=True)

    return df_rndeff_summary



#%% Define Variables
### ======================================
#computer name
# machine_name = platform.node().lower().replace('.','_')
machine_name = str( platform.node().lower().replace('.','_').replace('_cm_cluster', '') )
#time
exec_time = "%02.i-%02.i-%02.i_%02.i.%02.i.%02.i"%(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day,
                                                   datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second)

#region filter
reg2process = ('ALL')
#  = ('PW','SEE')
# reg2process = ('PW','SEE','JPN','TWN')
# reg2process = ('PW','SEE','CH','JPN','TWN')
# reg2process = ('PW','AK','SEE','CH','JPN','TWN')

#dPB scaling and offset
scl_dBP         = 0.01
rrup_offset_dBP = 50.

#parameters
#iteration samples
n_iter        = int(os.getenv('N_ITER')) if not os.getenv('N_ITER') is None else 200000
#multi-treading
flag_mthread  = bool(int(os.getenv('FLAG_MTHREAD'))) if not os.getenv('FLAG_MTHREAD') is None else True

#standard deviation penalty
lambda_std = float(os.getenv('LAMBDA_STD')) if not os.getenv('LAMBDA_STD') is None else 100.

#flag production
flag_production = bool(int(os.getenv('FLAG_PROD'))) if not os.getenv('FLAG_PROD') is None else True
#random realization
verif_rlz = int(os.getenv('VERIF_RLZ')) if not os.getenv('VERIF_RLZ') is None else 1

#flag regionalization
flag_regionalized = bool(int(os.getenv('FLAG_REG'))) if not os.getenv('FLAG_REG') is None else True

#flag heteroscedasticity
flag_heteroscedastic = bool(int(os.getenv('FLAG_HETERO'))) if not os.getenv('FLAG_HETERO') is None else True

#flag median vs mean value
flag_med = bool(int(os.getenv('FLAG_MED'))) if not os.getenv('FLAG_MED') is None else False

#frequency
freq = float(os.getenv('FREQ')) if not os.getenv('FREQ') is None else 1.2589
# freq = float(os.getenv('FREQ')) if not os.getenv('FREQ') is None else 1.0

#total residual column name
cname_dT  =  str(os.getenv('COLNAME_TOTRES'))  if not os.getenv('COLNAME_TOTRES')  is None else 'deltaT_med'
cname_dB  =  str(os.getenv('COLNAME_DELTAB'))  if not os.getenv('COLNAME_DELTAB')  is None else 'deltaB_med'
cname_dBP =  str(os.getenv('COLNAME_DELTABP')) if not os.getenv('COLNAME_DELTABP') is None else 'deltaBP_med'
cname_dS  =  str(os.getenv('COLNAME_DELTAS'))  if not os.getenv('COLNAME_DELTAS')  is None else 'deltaS_med'


#master filename
fname_main =  str(os.getenv('FILENAME_MASTER')) if not os.getenv('FILENAME_MASTER') is None else 'eas_f%.4fhz'%freq

#define master output filename
fname_main_out = fname_main + '_adj'

#stan model
dir_stan_model = '../../stan_lib/'
dir_stan_cmpl  = dir_stan_model  + 'compiled/%s_%s/'%(machine_name, exec_time)
fname_stan_model = 'partition_res_heteroscedastic_soft_constraint'

#flatfile directory    
if flag_production:
    dir_flt  = '../../../Data/gmm_ergodic/regression'
    dir_flt += '_regionalized' if flag_regionalized else '_global'
    dir_flt += '_heteroscedastic' if flag_heteroscedastic else '_homoscedastic'
    dir_flt += '_mthread/' if flag_mthread else '/'
else:
    dir_flt  = '../../../Data/gmm_ergodic/verification/regression/'
    dir_flt += 'updated_saturation/heteroscedastic_hangwall/'
    dir_flt += '_rlz%i/'%verif_rlz 
    
#filename coefficient and gm
fname_df_coeffs = "%s/%s_summary_coeff_"%(dir_flt,fname_main) + ("med" if flag_med else "mean") + ".csv"
fname_df_totres = "%s/%s_summary_gm.csv"%(dir_flt,fname_main)

#output directory
if flag_production:
    dir_out  = '../../../Data/gmm_ergodic/regression'
    dir_out += '_regionalized' if flag_regionalized else '_global'
    dir_out += '_heteroscedastic' if flag_heteroscedastic else '_homoscedastic'
    dir_out += '_mthread/' if flag_mthread else '/'
else:
    dir_out  = '../../../Data/gmm_ergodic/verification/regression/'
    dir_out += 'updated_saturation/heteroscedastic_hangwall/' 
    dir_out += '_rlz%i/'%verif_rlz 

#figures output
dir_fig = dir_out + 'figures/%s_partition_res/'%fname_main


#%% Read 
### ======================================
print("Partitioning residuals frequency: %.5fhz"%freq)

#hyper-parameters
df_coeffs = pd.read_csv(fname_df_coeffs, index_col=0)
#total residuals
df_totres = pd.read_csv(fname_df_totres)
#drop index column if present
if np.isin(['index'], df_totres.columns):
    df_totres.drop(columns=['index'], inplace=True)

#%% Processing Variables
### ======================================
#number of data points
n_data = len(df_totres)

if flag_regionalized:
    reg_namesall = np.unique(df_totres.reg)
    if not reg2process == 'ALL':
        reg_names = np.isin(reg_namesall, reg2process)
    else:
        reg_names = reg_namesall
else:
    reg_names = ['GLOBAL']


#%% Regression
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_stan_cmpl).mkdir(parents=True, exist_ok=True) 

#initialize list to collect dataframes with adjusted residuals
df_summary_adj_gm = list()

#compile stan model
if not os.path.isfile(dir_stan_cmpl+fname_stan_model):
    #copy regression file to compilation folder
    shutil.copy(dir_stan_model+fname_stan_model+'.stan', dir_stan_cmpl+fname_stan_model+'.stan')
    #compile stan model
    # if not flag_mthread:
    #     stan_model = cmdstanpy.CmdStanModel(stan_file=dir_stan_cmpl+fname_stan_model+'.stan', compile=True)
    # elif flag_mthread:
    #     cmdstanpy.set_cmdstan_path(os.path.join(os.getenv("HOME"),'.cmdstan/cmdstan-2.35.0-MT/'))
    #     stan_model = cmdstanpy.CmdStanModel(stan_file=dir_stan_cmpl+fname_stan_model+'.stan', 
    #                                         compile=True, cpp_options={'STAN_THREADS': 'TRUE'})
    print("Warning: Uncomment lines")
    stan_model = cmdstanpy.CmdStanModel(stan_file=dir_stan_cmpl+fname_stan_model+'.stan', compile=True)

#link file in new location
stan_model = cmdstanpy.CmdStanModel(stan_file=dir_stan_model+fname_stan_model+'.stan',
                                    exe_file=dir_stan_cmpl+fname_stan_model) 

#iterate over all regions
for reg_n in reg_names:
    print("\t region: %s"%reg_n)

    #regional dataframe
    df_totres_reg = df_totres.loc[df_totres.reg == reg_n,:] if flag_regionalized else df_totres
    
    #aleatory standard deviations
    s1 = df_coeffs.loc[reg_n,    's1']
    s2 = df_coeffs.loc['GLOBAL','s2']
    s3 = df_coeffs.loc[reg_n,    's3']
    s4 = df_coeffs.loc[reg_n,    's4']
    s5 = df_coeffs.loc[reg_n,    's5']
    s6 = df_coeffs.loc[reg_n,    's6']
    #magnitude breaks
    s1mag = df_coeffs.loc['GLOBAL','s1mag']
    s2mag = df_coeffs.loc['GLOBAL','s2mag']
    s5mag = df_coeffs.loc['GLOBAL','s5mag']
    s6mag = df_coeffs.loc['GLOBAL','s6mag']

    #compute adjusted residuals
    df_summary_adj_gm_reg = adjust_residuals(df_totres_reg, stan_model,
                                             s1, s2, s3, s4, s5, s6, s1mag, s2mag, s5mag, s6mag,
                                             scl_dBP=scl_dBP, rrup_offset_dBP=rrup_offset_dBP,
                                             cname_dT=cname_dT,
                                             cname_dB=cname_dB, cname_dBP=cname_dBP, cname_dS=cname_dS,
                                             n_iter=n_iter)
    
    df_summary_adj_gm.append(df_summary_adj_gm_reg)

#summarize adjusted residuals from all regions
df_summary_adj_gm = pd.concat(df_summary_adj_gm)
df_summary_adj_gm = df_summary_adj_gm.sort_values(by='motionid').reset_index(drop=True)

#identify unique events
_, eq_idx, eq_inv, eq_cnt = np.unique(df_summary_adj_gm[['eqid']].values, axis=0, 
                                      return_inverse=True, return_index=True, return_counts=True)
#create earthquake ids for all records (1 to n_eq)
eq_id = eq_inv + 1
n_eq = len(eq_idx)

#identify unique stations
_, st_idx, st_inv, st_cnt = np.unique(df_summary_adj_gm[['stid']].values, axis=0, 
                                      return_inverse=True, return_index=True, return_counts=True)
#create stationfor all records (1 to n_eq)
st_id = st_inv + 1
n_st = len(st_idx)

#metadata columns
c_eq_meta = ['eqid', 'eventid', 'reg', 'regid', 
             'eqlat', 'eqlon', 'eqx', 'eqy', 'eqz', 'eqcltlat', 'eqcltlon', 'eqcltx', 'eqclty', 'eqcltz', 'stlat', 
             'mag', 'ztor', 'sof', 'sofid', 'strike', 'dip', 'rake', 'eqclass']
c_st_meta = ['stid', 'stationid', 'reg', 'regid', 'stlat', 'stlon', 'stx', 'sty', 
             'vs30', 'vs30class', 'z1.0', 'z2.5', 'z1.0flag', 'z2.5flag']
#initiaize flatfile for sumamry of coefficient
df_summary_adj_eq = df_summary_adj_gm.loc[eq_idx,c_eq_meta+['deltaB','deltaBP','tau0','tauP']].reset_index(drop=True)
df_summary_adj_st = df_summary_adj_gm.loc[st_idx,c_st_meta+['deltaS','phiS']].reset_index(drop=True)
#add counts info
df_summary_adj_eq.loc[:,'eventcnt'] = eq_cnt
df_summary_adj_st.loc[:,'stationcnt'] = st_cnt
#move columns
print("move columns")

#delete compiled program
shutil.rmtree(dir_stan_cmpl)

f_dBP = df_totres.rrup.values - rrup_offset_dBP

dT_orig = df_totres.deltaT_med.values
dT_test = df_totres.deltaB_med.values + f_dBP * df_totres.deltaBP_med.values + df_totres.deltaS_med.values + df_totres.deltaWS_med.values  
dT_adj  = df_summary_adj_gm.deltaB.values + f_dBP * df_summary_adj_gm.deltaBP.values + df_summary_adj_gm.deltaS.values + df_summary_adj_gm.deltaWS.values  


# assert(False)
# df_summary_adj_gm[['deltaWS']] *= -1.
# df_summary_adj_eq[['deltaB','deltaBP']] *= -1.
# df_summary_adj_st[['deltaS']] *= -1.


#%% Output
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#save ground motion and random effects
fname_df = (fname_main_out + '_summary_gm').replace(' ','_')
df_summary_adj_gm.to_csv(dir_out + fname_df + '.csv', index=False)
fname_df = (fname_main_out + '_summary_eq').replace(' ','_')
df_summary_adj_eq.to_csv(dir_out + fname_df + '.csv', index=False)
fname_df = (fname_main_out + '_summary_st').replace(' ','_')
df_summary_adj_st.to_csv(dir_out + fname_df + '.csv', index=False)

#%% Plotting
### ======================================
figures_residuals(df_summary_adj_gm, df_summary_adj_eq, df_summary_adj_st,
                  cn_dB='deltaB', cn_dBP='deltaBP', 
                  cn_dS='deltaS', cn_dWS='deltaWS', 
                  scl_dBP=scl_dBP,
                  dir_fig=dir_fig, fname_main_out=fname_main_out)

figures_residuals_adjust(df_summary_adj_gm, df_summary_adj_eq, df_summary_adj_st,
                         df_totres,
                         df_totres.loc[eq_idx,c_eq_meta+['deltaB_med','deltaB_mu','deltaB_sd','deltaBP_med','deltaBP_mu','deltaBP_sd']].reset_index(drop=True),
                         df_totres.loc[st_idx,c_st_meta+['deltaS_med','deltaS_mu','deltaS_sd']].reset_index(drop=True),
                         cn_dB=['deltaB','deltaB_mu','deltaB_sd'], cn_dBP=['deltaBP','deltaBP_mu','deltaBP_sd'], 
                         cn_dS=['deltaS','deltaS_mu','deltaS_sd'], cn_dWS=['deltaWS','deltaWS_mu','deltaWS_sd'], 
                         scl_dBP=scl_dBP,
                         resid_range=[-3., 3.],
                         dir_fig=dir_fig, fname_main_out=fname_main_out, flag_synthetic=False)

