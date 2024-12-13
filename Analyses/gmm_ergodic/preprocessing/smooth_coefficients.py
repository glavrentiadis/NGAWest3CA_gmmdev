#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:43:34 2024

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
from pylib_general import gaussian_convolution_nonuniform

#%% Define Variables
### ======================================
#filename original gmm coefficients
fn_coeffs_orig = '../../../Raw_files/model_coeffs/coeff_20241021_mod4.csv'

#smoothing kernel width
sig_all = 0.25
sig_coeff = {'c6':1.5}

#output directory
dir_out = '../../../Data/gmm_ergodic/preprocessing/'
dir_fig = dir_out + 'figures/'

#%% Read Data
### ======================================
#read original coefficients
df_coeffs_orig = pd.read_csv(fn_coeffs_orig)

#%% Processing Variables
### ======================================
#frequency array
freq_array = df_coeffs_orig.f.values

#coefficient names
coeff_names = df_coeffs_orig.columns
coeff_names = coeff_names[~np.isin(coeff_names, 'f')]

#change coefficients to floating points
df_coeffs_orig[coeff_names] = df_coeffs_orig[coeff_names].astype(float)

#smoothed coefficients dataframe
df_coeffs_smooth = df_coeffs_orig.copy()

#smooth coefficients
for j, c_n in enumerate(coeff_names):
    #smoothing kernel
    sig = sig_coeff[c_n] if np.isin(c_n, list(sig_coeff.keys())) else sig_all
    
    #original coefficients
    coef_array_orig   = df_coeffs_orig.loc[:,c_n].values
    #smoothed coefficients
    coef_array_smooth = gaussian_convolution_nonuniform(np.log(freq_array), coef_array_orig, sig)
    
    #store smoothed coefficients
    df_coeffs_smooth.loc[:,c_n] = coef_array_smooth


#%% Output
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#summarize coefficients
fname_coeffs_orig   = 'eas_coefs_orig'
df_coeffs_orig.to_csv(dir_out + fname_coeffs_orig + '.csv', index=False)
fname_coeffs_smooth = 'eas_coefs_smoothed'
df_coeffs_smooth.to_csv(dir_out + fname_coeffs_smooth + '.csv', index=False)

#%% Plotting
### ======================================
#x-axis labels
labels = {'c1':'$c_1$', 'c2':'$c_2$', 'c3':'$c_3$', 'c4':'$c_4$', 'c5':'$c_5$',
          'c6':'$c_6$', 'c7':'$c_7$', 'c8':'$c_8$', 'c9':'$c_9$',
          'c10a':'$c_{10 \\alpha}$', 'c10b':'$c_{10 \\beta}$', 
          'c11': '$c_{11}$', 'c13': '$c_{13}$', 
          'chm': '$c_{hm}$', 'cmag': '$c_{M}$', 'cn': '$c_{n}$', 
          's1':'$s_1$', 's2':'$s_2$', 's3':'$s_3$', 's4':'$s_4$', 's5':'$s_5$', 's6':'$s_6$', 
          's1M': '$s_{1M}$', 's2M': '$s_{2M}$', 's5M': '$s_{5M}$', 's6M': '$s_{6M}$'}

#iterate over coefficients
for c_n in coeff_names:
    
    #figure name and axes
    fname_fig = ('eas_summary_' + c_n).replace(' ','_')
    fig, ax = plt.subplots(figsize = (20,20), nrows=1, ncols=1)
    #plot original and smoothed coefficinet vs freq
    hl = ax.semilogx(freq_array, df_coeffs_smooth.loc[:,c_n], '-', color='black', linewidth=3, 
                     label='Smoothed')
    hl = ax.semilogx(freq_array, df_coeffs_orig.loc[:,c_n], '--', color='black', linewidth=3, 
                     label='Original')
    #edit properties
    ax.set_xlabel(r'frequency (hz)', fontsize=32)
    ax.set_ylabel(labels[c_n] if np.isin(c_n, list(labels.keys())) else c_n, fontsize=32)
    ax.legend(loc='lower right', fontsize=30)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    
    #save figure
    fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')
    plt.close(fig)

