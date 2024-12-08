#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:33:05 2024

@author: glavrent
"""

#load libraries
#general
import sys
import pathlib
import warnings
#arithmetic libraries
import numpy as np
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick
#user libraries
sys.path.insert(0,'../../python_lib')
from pylib_general import combine_dataframes

#%% Define Variables
### ======================================
#frequency array
freq_array = np.array([0.013182570, 0.015135620, 0.019952621, 0.025118870, 0.030199520, 0.039810720, 0.050118730,  0.060255960, 0.075857751, 0.079432822, 0.095499262, 
                       0.100000000,              0.151356110,              0.199526200, 0.251188600,                            0.301995200,              0.398107140, 0.501187200, 0.602559500, 0.758577600,
                       1.000000000, 1.258926000, 1.513561000,              1.995262100, 2.511886300,                            3.019952000,              3.981071000, 5.011872000, 6.025596000, 7.585776000, 8.511379200,
                       10.00000000, 12.02264200, 15.13561400, 16.98244000, 19.95262100, 21.87761100, 25.118860000, 27.54229100, 30.19952000, 35.48133400, 39.81071000])

#flag to keep only global coefficients
flag_glob = False

#include prior distribution
flag_prior = True
fn_coeffs_prior = '../../../Raw_files/model_coeffs/coeff_20241021_mod3.csv'

#data directory
# dir_data = '../../../Data/gmm_ergodic/regression/'
# dir_data = '../../../Data/gmm_ergodic/regression_mthread/'
dir_data = '../../../Data/gmm_ergodic/regression_wo_smag_mthread/'

#use median vs mean value
flag_med = False

#output directory
dir_out = dir_data
#figures output
dir_fig  = dir_data + 'figures/eas_summary/'

#%% Read Data
### ======================================
#initalize frequencies and coefficient list
freq_list = list()
df_coeff_freqs = list()
#initalize coefficient and region names
coeff_names = list()
reg_names   = list() 

#iterate over frequencies and read coefficients
for j, freq in enumerate(freq_array):

    #read coefficients
    try:
        fname_df_coeff_freq = ('eas_f%.4fhz_summary_coeff'%freq + ('_med' if flag_med else '_mean') + '.csv').replace(' ','_')
        df_coeff_freqs.append( pd.read_csv(dir_data + fname_df_coeff_freq, index_col=0) )
    except FileNotFoundError:
        warnings.warn("Unavailable frequency: %.4fhz"%freq)
        continue

    #collect frequencies
    freq_list.append( freq )
    
    #collect coefficient and region names
    coeff_names.append( df_coeff_freqs[-1].columns )
    reg_names.append( df_coeff_freqs[-1].index )
    
#keep only unique entries
coeff_names = np.unique(np.hstack(coeff_names))
reg_names   = np.unique(np.hstack(reg_names))
#coefficient names
coeff_names = np.delete(coeff_names, np.argwhere(np.isin(coeff_names,['s1mag','s2mag','s5mag','s6mag'])))

#read prior coefficients
df_coeffs_prior = pd.read_csv(fn_coeffs_prior)

#%% Processing
### ======================================
#summarize coefficients from all frequencies
if flag_glob:
    df_coeff_freqs = pd.concat([df_c.loc['GLOBAL',:] for df_c in df_coeff_freqs], axis=0)
else:
    df_coeff_freqs = combine_dataframes(df_coeff_freqs)

#add frequency information
df_coeff_freqs.insert(0, 'freq', freq_list)

#%% Output
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#summarize coefficients
fname_coeffs = 'eas_summary_coefs' + ('_med' if flag_med else '_mean')
df_coeff_freqs.to_csv(dir_data + fname_coeffs + '.csv', index=False)

#%% Plotting
### ======================================
#color map
cmap = plt.get_cmap("tab10")
#x-axis labels
labels = {'c1':'$c_1$', 'c2':'$c_2$', 'c3':'$c_3$', 'c4':'$c_4$', 'c5':'$c_5$',
          'c6':'$c_6$', 'c7':'$c_7$', 'c8':'$c_8$', 'c9':'$c_9$',
          'c10a':'$c_{10 \\alpha}$', 'c10b':'$c_{10 \\beta}$', 
          'c11': '$c_{11}$', 'c13': '$c_{13}$', 
          'chm': '$c_{hm}$', 'cmag': '$c_{M}$', 'cn': '$c_{n}$', 
          's1':'$s_1$', 's2':'$s_2$', 's3':'$s_3$', 's4':'$s_4$', 's5':'$s_5$', 's6':'$s_6$', 
          's1mag': '$s_{1M}$', 's2mag': '$s_{2M}$', 's5mag': '$s_{5M}$', 's6mag': '$s_{6M}$'}

#iterate over coefficients
for c_n in coeff_names:
    
    #figure name and axes
    fname_fig = ('eas_summary_' +  ('med_' if flag_med else 'mean_') + c_n).replace(' ','_')
    fig, ax = plt.subplots(figsize = (20,20), nrows=1, ncols=1)
    #plot global and regional coefficinets
    for j, r_n in enumerate(reg_names):
        if np.isin(f"{c_n}_{r_n}", df_coeff_freqs.columns):
            #line properties
            clr = 'black'          if r_n.upper() == 'GLOBAL' else cmap(j-1) #line color
            lw  = 3.               if r_n.upper() == 'GLOBAL' else 2         #linewidth
            lb  = r_n.capitalize() if r_n.upper() == 'GLOBAL' else r_n
            zo  = 2 if r_n.upper() == 'GLOBAL' else 1
            #frequency versus freq
            hl = ax.semilogx(freq_list, df_coeff_freqs.loc[:,f"{c_n}_{r_n}"],
                             '-o', color=clr, linewidth=lw, 
                             label=lb, zorder=zo)
        else:
            continue
    
    #plot prior mean
    if flag_prior and np.isin(c_n, df_coeffs_prior.columns):
        hl = ax.semilogx(df_coeffs_prior.f, df_coeffs_prior.loc[:,f"{c_n}"],
                         '--', color='black', linewidth=1.5, 
                         label='Prior mean', zorder=1)
    #edit properties
    ax.set_xlabel(r'frequency (hz)', fontsize=32)
    ax.set_ylabel(labels[c_n] if np.isin(c_n, list(labels.keys())) else c_n, fontsize=32)
    if not flag_glob: ax.legend(loc='lower right', fontsize=30)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    
    #save figure
    fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')
    plt.close(fig)

