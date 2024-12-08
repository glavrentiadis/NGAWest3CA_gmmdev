#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:21:49 2024

@author: glavrent
"""

#load libraries
#general
import os
import sys
import pathlib
#arithmetic libraries
import numpy as np
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick
#ground motion libraries
import pygmm
#user libraries
sys.path.insert(0,'../../python_lib')
from pylib_gmm import gmm_eas_upd
from pylib_gmm import scalingHW as scaling_hw 
#user functions
def parse_coeff(df_coeff, flag_reg, glob='GLOBAL', reg=['PW', 'SEE', 'TWN', 'JPN']):
    
    if flag_reg and np.all(~np.isnan(df_coeff.values)):
        return df_coeff.loc[reg].values
    else:
        return df_coeff.loc[glob]

#%% Define Variables
### ======================================
#define scenario
#source info
mag   = float(os.getenv('MAG'))   if not os.getenv('MAG')   is None else 7.0
width = float(os.getenv('WIDTH')) if not os.getenv('WIDTH') is None else 20
dip   = float(os.getenv('DIP'))   if not os.getenv('DIP')   is None else 90.
ztor  = float(os.getenv('ZTOR'))  if not os.getenv('ZTOR')  is None else 2.
sof   = float(os.getenv('SOF'))   if not os.getenv('SOF')   is None else 'SS'
#path info
rjb   = float(os.getenv('RJB'))  if not os.getenv('RJB')  is None else 25.
rrup  = float(os.getenv('RRUP')) if not os.getenv('RRUP') is None else 25.
rx    = float(os.getenv('RX'))   if not os.getenv('RX')   is None else 25.
ry0   = float(os.getenv('RY0'))  if not os.getenv('RY0')  is None else 0.
#site info
vs30 = float(os.getenv('VS30')) if not os.getenv('VS30') is None else 500.
z1 = None

#use median vs mean value
flag_med = True

#frequency array
freq_array = np.array([0.013182570, 0.015135620, 0.019952621, 0.025118870, 0.030199520, 0.039810720, 0.050118730,  0.060255960, 0.075857751, 0.079432822, 0.095499262, 
                       0.100000000,              0.151356110,              0.199526200, 0.251188600,                            0.301995200,              0.398107140, 0.501187200, 0.602559500, 0.758577600,
                       1.000000000, 1.258926000, 1.513561000,              1.995262100, 2.511886300,                            3.019952000,              3.981071000, 5.011872000, 6.025596000, 7.585776000, 8.511379200,
                       10.00000000, 12.02264200, 15.13561400, 16.98244000, 19.95262100, 21.87761100, 25.118860000, 27.54229100, 30.19952000, 35.48133400, 39.81071000])

#coefficient directory
dir_coeffs = '../../../Data/gmm_ergodic/regression_prev/'

#output directory
dir_out = '../../../Data/gmm_ergodic/prediction/scenarios/'
#figures output
dir_fig = dir_out + 'figures/'

#%% Read Data
### ======================================
#initalize coefficient dictionary for all frequencies
df_coeff_freqs = dict()
freq_list = list()

#iterate over frequencies and read coefficients
for j, freq in enumerate(freq_array):
    #read coefficients
    try:
        fname_df_coeff_freq = ('eas_f%.4fhz_summary_coeff'%freq + '.csv').replace(' ','_')
        df_coeff_freqs['eas%.4fhz'%freq] = pd.read_csv(dir_coeffs + fname_df_coeff_freq, index_col=0)
    except FileNotFoundError:
        continue

    #collect frequencies
    freq_list.append(freq)
    
    
#gmm regions
regions = np.unique( df_coeff_freqs['eas%.4fhz'%freq_list[0]].index ).tolist()
# regions = df_coeff_freqs[k_f].index[~np.isin(df_coeff_freqs[k_f].index, 'GLOBAL')].to_list()
    
#%% Processing
### ======================================
#number of frequencies
n_f = len(freq_list)

#initialize arrays
f_med = np.full((n_f, len(regions)), np.nan)
tau0  = np.full((n_f, len(regions)), np.nan)
tauP  = np.full((n_f, len(regions)), np.nan)
phiS  = np.full((n_f, len(regions)), np.nan)
phi0  = np.full((n_f, len(regions)), np.nan)

#compute median ground moton
for j, (f, k_f) in enumerate(zip(freq_list, df_coeff_freqs)):
   
    #linear coefficients
    c1    = parse_coeff(df_coeff_freqs[k_f].loc[:,'c1'],   True, reg=regions)
    c2    = parse_coeff(df_coeff_freqs[k_f].loc[:,'c2'],   True, reg=regions)
    c3    = parse_coeff(df_coeff_freqs[k_f].loc[:,'c3'],   True, reg=regions)
    c4    = parse_coeff(df_coeff_freqs[k_f].loc[:,'c4'],   True, reg=regions)
    c7    = parse_coeff(df_coeff_freqs[k_f].loc[:,'c7'],   True, reg=regions)
    c8    = parse_coeff(df_coeff_freqs[k_f].loc[:,'c8'],   True, reg=regions)
    c9    = parse_coeff(df_coeff_freqs[k_f].loc[:,'c9'],   True, reg=regions)
    c10a  = parse_coeff(df_coeff_freqs[k_f].loc[:,'c10a'], True, reg=regions)
    c10b  = parse_coeff(df_coeff_freqs[k_f].loc[:,'c10b'], True, reg=regions)
    c11   = parse_coeff(df_coeff_freqs[k_f].loc[:,'c11'],  True, reg=regions)
    c13   = parse_coeff(df_coeff_freqs[k_f].loc[:,'c13'],  True, reg=regions)
 
    #non-linear coefficients
    c5    = parse_coeff(df_coeff_freqs[k_f].loc[:,'c5'],   True, reg=regions)
    c6    = parse_coeff(df_coeff_freqs[k_f].loc[:,'c6'],   True, reg=regions)
    chm   = parse_coeff(df_coeff_freqs[k_f].loc[:,'chm'],  True, reg=regions)
    cn    = parse_coeff(df_coeff_freqs[k_f].loc[:,'cn'],   True, reg=regions)
    cmag  = parse_coeff(df_coeff_freqs[k_f].loc[:,'cmag'], True, reg=regions)

    #aleatory variability
    s1    = parse_coeff(df_coeff_freqs[k_f].loc[:,'s1'],   True, reg=regions)
    s2    = parse_coeff(df_coeff_freqs[k_f].loc[:,'s2'],   True, reg=regions)
    s3    = parse_coeff(df_coeff_freqs[k_f].loc[:,'s3'],   True, reg=regions)
    s4    = parse_coeff(df_coeff_freqs[k_f].loc[:,'s4'],   True, reg=regions)
    s5    = parse_coeff(df_coeff_freqs[k_f].loc[:,'s5'],   True, reg=regions)
    s6    = parse_coeff(df_coeff_freqs[k_f].loc[:,'s6'],   True, reg=regions)
    #aleatory variability (mag break)
    s1mag = parse_coeff(df_coeff_freqs[k_f].loc[:,'s1mag'], True, reg=regions)
    s2mag = parse_coeff(df_coeff_freqs[k_f].loc[:,'s2mag'], True, reg=regions)
    s5mag = parse_coeff(df_coeff_freqs[k_f].loc[:,'s5mag'], True, reg=regions)
    s6mag = parse_coeff(df_coeff_freqs[k_f].loc[:,'s6mag'], True, reg=regions)
    
    #iterate over all regions
    for k, reg in enumerate(regions):
        #compute ground motion
        out = gmm_eas_upd(mag, ztor, sof, rrup, vs30, z1, 
                         reg=reg,
                         regions=regions,
                         c1=c1, c2=c2, c3=c3, c4=c4, 
                         c5=c5, c6=c6, c7=c7, c8=c8, c9=c9,
                         c10a=c10a, c10b=c10b, 
                         c11=c11, z1p0_vs30_brk=[],
                         cn=cn, cmag=cmag, chm=chm, ztor_max=20.,
                         s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6,
                         s1mag=s1mag, s2mag=s2mag, s5mag=s5mag, s6mag=s6mag)
        #parse gmm
        f_med[j,k] = out[0].item()
        tau0[j,k]  = out[4].item()
        tauP[j,k]  = out[5].item()
        phiS[j,k]  = out[6].item()
        phi0[j,k]  = out[7].item()
        
        
        #hanging all scaing
        f_hw = scaling_hw(mag, width, dip, ztor, rjb, rrup, rx, ry0)
        #update median ground motion
        f_med[j,k] += c13 * f_hw
    
#evaluate BA18
gmm_scen = pygmm.Scenario(mag=mag, dip=dip, mechanism=sof, depth_tor=ztor, 
                          dist_rup=rrup, 
                          v_s30=vs30)


#define eas gmm
gmm_ba19 = pygmm.BaylessAbrahamson2019(gmm_scen)

#%% Output
### ======================================
#create output directory
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#color map
cmap = plt.get_cmap("tab10")

#figure name and axes
fname_fig = 'gmm_eas_cmp_shape_M_%.1f_rrup_%.1fkm_vs30_%.1fmsec_regional'%(mag,rrup,vs30)
fig, ax = plt.subplots(figsize = (20,20), nrows=1, ncols=1)
#plot new and BA19 gmms
for k, reg in enumerate(regions):
    #line properties
    clr = 'black'          if reg == 'GLOBAL' else cmap(k-1) #line color
    lw  = 3.               if reg == 'GLOBAL' else 3.         #linewidth
    lb  = reg.capitalize() if reg == 'GLOBAL' else reg
    zo  = 2 if reg == 'GLOBAL' else 1
    #new gmm
    hl_new  = ax.loglog(freq_list, np.exp(f_med[:,k]), '-',  linewidth=lw, color=clr, label=lb, zorder=zo)
#ba19
hl_ba19 = ax.loglog(gmm_ba19.freqs, gmm_ba19.eas, '--', linewidth=2., color='black', label='BA19')
#edit properties
ax.set_xlabel(r'frequency (hz)', fontsize=32)
ax.set_ylabel('EAS [g-sec]', fontsize=32)
ax.set_title('Scenario: $M$%.1f $R_{rup}$=%.1fkm $V_{S30}$=%.1fm/sec'%(mag,rrup,vs30), fontsize=32)
ax.legend(loc='upper right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
#save figure
fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')
plt.close(fig)