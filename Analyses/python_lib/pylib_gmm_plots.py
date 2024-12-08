#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:52:56 2024

@author: glavrent
"""
#load libraries
#general
import os
import sys
import shutil
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
#user libraries
sys.path.insert(0,'./python_lib')
from moving_mean import movingmean
import pylib_contour_plots as pycplt


def figures_residuals(df_summary_gm, df_summary_eq, df_summary_st,
                      cn_dB, cn_dBP, cn_dS, cn_dWS, scl_dBP=1.,
                      dir_fig='./', fname_main_out='gmm_residuals'):

    #color map
    cmap = plt.get_cmap("tab10")

    #number of groundmotions, events, and stations
    n_gm = len(df_summary_gm)
    n_eq = len(df_summary_eq)
    n_st = len(df_summary_st)
    
    for j, regid in enumerate( np.insert(np.unique(df_summary_eq.regid), 0, 0) ):
    
        #region points
        if regid == 0:
            i_reg_gm = np.full(n_gm, True)
            i_reg_eq = np.full(n_eq, True)
            i_reg_st = np.full(n_st, True)
            #region name
            reg_name = 'ALL'
        else:
            i_reg_gm = df_summary_gm.regid == regid
            i_reg_eq = df_summary_eq.regid == regid
            i_reg_st = df_summary_st.regid == regid
            #region name
            reg_name = df_summary_gm.loc[i_reg_gm,'reg'].unique()[0]
        
        #region-specific dataframe
        df_summary_gm_reg = df_summary_gm.loc[i_reg_gm,:].reset_index(drop=True) 
        df_summary_eq_reg = df_summary_eq.loc[i_reg_eq,:].reset_index(drop=True)
        df_summary_st_reg = df_summary_st.loc[i_reg_st,:].reset_index(drop=True)
        
        #magnitude bins
        mag_bins = np.arange(3, 8.6, 0.5)
        mag_dBbin, _, mag_dBmu, _, mag_dB16prc, mag_dB84prc = movingmean(df_summary_eq_reg[cn_dB],
                                                                         df_summary_eq_reg.mag, mag_bins)
        mag_dBPbin, _, mag_dBPmu, _, mag_dBP16prc, mag_dBP84prc = movingmean(df_summary_eq_reg[cn_dBP],
                                                                             df_summary_eq_reg.mag, mag_bins)
        mag_dWSbin, _, mag_dWSmu, _, mag_dWS16prc, mag_dWS84prc = movingmean(df_summary_gm_reg[cn_dWS],
                                                                             df_summary_gm_reg.mag, mag_bins)
    
        #rupture distance bins
        rrup_bins = np.logspace(np.log10(1), np.log10(400), 6)
        rrup_dWSbin, _, rrup_dWSmu, _, rrup_dWS16prc, rrup_dWS84prc = movingmean(df_summary_gm_reg[cn_dWS],
                                                                                 df_summary_gm_reg.rrup, rrup_bins)
        
        #vs30 bins
        vs30_bins = np.logspace(np.log10(100), np.log10(2000), 6)
        vs30_dSbin, _, vs30_dSmu, _, vs30_dS16prc, vs30_dS84prc = movingmean(df_summary_st_reg[cn_dS],
                                                                             df_summary_st_reg.vs30, vs30_bins)
        vs30_dWSbin, _, vs30_dWSmu, _, vs30_dWS16prc, vs30_dWS84prc = movingmean(df_summary_gm_reg[cn_dWS],
                                                                                 df_summary_gm_reg.vs30, vs30_bins)
    
    
        #between-event residuals
        fname_fig = (fname_main_out + '_deltaB_scl').replace(' ','_')
        if not regid==0: fname_fig += '_reg_%s'%reg_name
        fig, ax = plt.subplots(figsize = (20,10), nrows=1, ncols=1)
        for j, rid in enumerate(np.unique(df_summary_eq_reg.regid)):
            #region points
            i_r = df_summary_eq_reg.regid == rid
            #region name
            rn = df_summary_eq_reg.loc[i_r,'reg'].unique()[0]
            #region plot handle
            hl = ax.plot(df_summary_eq_reg.loc[i_r,'mag'], df_summary_eq_reg.loc[i_r,cn_dB], 
                         'o', color=cmap(j), label=rn)
        hl = ax.plot(mag_dBbin, mag_dBmu, 's', color='black', label='Mean')
        hl = ax.errorbar(mag_dBbin, mag_dBmu, yerr=np.abs(np.vstack((mag_dB16prc,mag_dB84prc)) - mag_dBmu),
                          capsize=8, fmt='none', ecolor='black', linewidth=2,
                          label=r'$16-84^{th}$'+'\n Percentile')
        #edit properties
        ax.set_xlabel(r'Magnitude', fontsize=32)
        ax.set_ylabel(r'$\delta B$', fontsize=32)
        if regid==0: ax.legend(loc='lower right', fontsize=30, ncols=2)
        ax.grid(which='both')
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        ax.set_ylim([-3, 3])
        ax.set_yticks([-3.,-1.5,0.,1.5,3.])
        fig.tight_layout()
        fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')
        plt.close(fig)
        
        #between-event-path residuals
        fname_fig = (fname_main_out + '_deltaBP_scl').replace(' ','_')
        if not regid==0: fname_fig += '_reg_%s'%reg_name
        fig, ax = plt.subplots(figsize = (20,10), nrows=1, ncols=1)
        for j, rid in enumerate(np.unique(df_summary_eq_reg.regid)):
            #region points
            i_r = df_summary_eq_reg.regid == rid
            #region name
            rn = df_summary_eq_reg.loc[i_r,'reg'].unique()[0]
            #region plot handle
            hl = ax.plot(df_summary_eq_reg.loc[i_r,'mag'], 1/scl_dBP*df_summary_eq_reg.loc[i_r,cn_dBP], 
                         'o', color=cmap(j), label=rn)
        hl = ax.plot(mag_dBPbin, 1/scl_dBP*mag_dBPmu, 's', color='black', label='Mean')
        hl = ax.errorbar(mag_dBPbin, 1/scl_dBP*mag_dBPmu, 
                         yerr=1/scl_dBP*np.abs(np.vstack((mag_dBP16prc,mag_dBP84prc)) - mag_dBPmu),
                         capsize=8, fmt='none', ecolor='black', linewidth=2,
                         label=r'$16-84^{th}$'+'\n Percentile')
        #edit properties
        ax.set_xlabel(r'Magnitude', fontsize=32)
        ax.set_ylabel(r'$\delta BP$ (scaled)', fontsize=32)
        if regid==0: ax.legend(loc='lower right', fontsize=30, ncols=2)
        ax.grid(which='both')
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        # ax.set_ylim([-3, 3])
        # ax.set_yticks([-3.,-1.5,0,1.5,3.])
        ax.set_ylim([-1.5, 1.5])
        ax.set_yticks([-1.5,-0.75,0.,0.75,1.5])
        fig.tight_layout()
        fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')
        plt.close(fig)
        
        #between-site residuals
        fname_fig = (fname_main_out + '_deltaS_scl').replace(' ','_')
        if not regid==0: fname_fig += '_reg_%s'%reg_name
        fig, ax = plt.subplots(figsize = (20,10), nrows=1, ncols=1)
        for j, rid in enumerate(np.unique(df_summary_st_reg.regid)):
            #region points
            i_r = df_summary_st_reg.regid == rid
            #region name
            rn = df_summary_st_reg.loc[i_r,'reg'].unique()[0]
            #region plot handle
            hl = ax.semilogx(df_summary_st_reg.loc[i_r,'vs30'], df_summary_st_reg.loc[i_r,cn_dS], 
                             'o', color=cmap(j), label=rn)
        hl = ax.plot(vs30_dSbin, vs30_dSmu, 's', color='black', label='Mean')
        hl = ax.errorbar(vs30_dSbin, vs30_dSmu, yerr=np.abs(np.vstack((vs30_dS16prc,vs30_dS84prc)) - vs30_dSmu),
                         capsize=8, fmt='none', ecolor='black', linewidth=2,
                         label=r'$16-84^{th}$'+'\n Percentile')
        #edit properties
        ax.set_xlabel(r'$V_{S30}$',   fontsize=32)
        ax.set_ylabel(r'$\delta S$', fontsize=32)
        if regid==0: ax.legend(loc='lower right', fontsize=30, ncols=2)
        ax.grid(which='both')
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        ax.set_xlim([100, 2500])
        ax.set_ylim([-3, 3])
        ax.set_yticks([-3.,-1.5,0.,1.5,3.])
        fig.tight_layout()
        fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')
        plt.close(fig)
        
        #within-event-site
        fname_fig = (fname_main_out + '_deltaWS_scl').replace(' ','_')
        if not regid==0: fname_fig += '_reg_%s'%reg_name
        #within-event-site (mag scaling)
        fig, ax = plt.subplots(figsize = (20,3*10), nrows=3, ncols=1)
        for j, rid in enumerate(np.unique(df_summary_gm_reg.regid)):
            #region points
            i_r = df_summary_gm_reg.regid == rid
            #region name
            rn = df_summary_gm_reg.loc[i_r,'reg'].unique()[0]
            #region plot handle
            hl = ax[0].plot(df_summary_gm_reg.loc[i_r,'mag'], df_summary_gm_reg.loc[i_r,cn_dWS], 
                            'o', color=cmap(j), label=rn)
        hl = ax[0].plot(mag_dWSbin, mag_dWSmu, 's', color='black', label='Mean')
        hl = ax[0].errorbar(mag_dWSbin, mag_dWSmu, yerr=np.abs(np.vstack((mag_dWS16prc,mag_dWS84prc)) - mag_dWSmu),
                            capsize=8, fmt='none', ecolor='black', linewidth=2,
                            label=r'$16-84^{th}$'+'\n Percentile')
        #edit properties
        ax[0].set_xlabel(r'Magnitude',   fontsize=32)
        ax[0].set_ylabel(r'$\delta WS$', fontsize=32)
        if regid==0: ax[0].legend(loc='lower right', fontsize=30, ncols=2)
        ax[0].grid(which='both')
        ax[0].tick_params(axis='x', labelsize=30)
        ax[0].tick_params(axis='y', labelsize=30)
        # ax[0].set_ylim([-3, 3])
        ax[0].set_ylim([-4, 4])
        # ax[0].set_yticks([-3.,-1.5,0.,1.5,3.])
        ax[0].set_yticks([-4.,-2.,0.,2.,4.])
        #within-event-site (rrup scaling)
        for j, rid in enumerate(np.unique(df_summary_gm_reg.regid)):
            #region points
            i_r = df_summary_gm_reg.regid == rid
            #region name
            rn = df_summary_gm_reg.loc[i_r,'reg'].unique()[0]
            #region plot handle
            hl = ax[1].semilogx(df_summary_gm_reg.loc[i_r,'rrup'], df_summary_gm_reg.loc[i_r,cn_dWS], 
                             'o', color=cmap(j), label=rn)
        hl = ax[1].semilogx(rrup_dWSbin, rrup_dWSmu, 's', color='black', label='Mean')
        hl = ax[1].errorbar(rrup_dWSbin, rrup_dWSmu, yerr=np.abs(np.vstack((rrup_dWS16prc,rrup_dWS84prc)) - rrup_dWSmu),
                            capsize=8, fmt='none', ecolor='black', linewidth=2,
                            label=r'$16-84^{th}$'+'\n Percentile')
        #edit properties
        ax[1].set_xlabel(r'$R_{rup}$ km', fontsize=32)
        ax[1].set_ylabel(r'$\delta WS$',  fontsize=32)
        #if regid==0: ax[1].legend(loc='lower right', fontsize=30)
        ax[1].grid(which='both')
        ax[1].tick_params(axis='x', labelsize=30)
        ax[1].tick_params(axis='y', labelsize=30)
        # ax[1].set_ylim([-3, 3])
        ax[1].set_ylim([-4, 4])
        # ax[1].set_yticks([-3.,-1.5,0.,1.5,3.])
        ax[1].set_yticks([-4.,-2.,0.,2.,4.])
        #within-event-site (vs30 scaling)
        for j, rid in enumerate(np.unique(df_summary_gm_reg.regid)):
            #region points
            i_r = df_summary_gm_reg.regid == rid
            #region name
            rn = df_summary_gm_reg.loc[i_r,'reg'].unique()[0]
            #region plot handle
            hl = ax[2].semilogx(df_summary_gm_reg.loc[i_r,'vs30'], df_summary_gm_reg.loc[i_r,cn_dWS], 
                             'o', color=cmap(j), label=rn)
        hl = ax[2].plot(vs30_dWSbin, vs30_dWSmu, 's', color='black', label='Mean')
        hl = ax[2].errorbar(vs30_dWSbin, vs30_dWSmu, yerr=np.abs(np.vstack((vs30_dWS16prc,vs30_dWS84prc)) - vs30_dWSmu),
                            capsize=8, fmt='none', ecolor='black', linewidth=2,
                            label=r'$16-84^{th}$'+'\n Percentile')
        #edit properties
        ax[2].set_xlabel(r'$V_{S30}$',   fontsize=32)
        ax[2].set_ylabel(r'$\delta WS$', fontsize=32)
        #if regid==0: ax[2].legend(loc='lower right', fontsize=30)
        ax[2].grid(which='both')
        ax[2].tick_params(axis='x', labelsize=30)
        ax[2].tick_params(axis='y', labelsize=30)
        ax[2].set_xlim([100, 2500])
        ax[2].set_ylim([-3, 3])
        # ax[2].set_ylim([-4, 4])
        # ax[2].set_yticks([-3.,-1.5,0.,1.5,3.])
        ax[2].set_yticks([-4.,-2.,0.,2.,4.])
        fig.tight_layout()
        fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')
        plt.close(fig)
    
    #between-event residuals (spatial variability)
    for j, rid in enumerate(np.unique(df_summary_eq.regid)):
        #region points
        i_r = df_summary_eq.regid == rid
        #region name
        rn = df_summary_eq.loc[i_r,'reg'].unique()[0]
        #figure name
        fname_fig = (fname_main_out + '_deltaB_reg_' + rn).replace(' ','_')
        #deltaB
        data2plot = df_summary_eq.loc[i_r,['eqlat','eqlon',cn_dB]].values
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
        plt.close(fig)
    
    #between-event-path residuals (spatial variability)
    for j, rid in enumerate(np.unique(df_summary_eq.regid)):
        #region points
        i_r = df_summary_eq.regid == rid
        #region name
        rn = df_summary_eq.loc[i_r,'reg'].unique()[0]
        #figure name
        fname_fig = (fname_main_out + '_deltaBP_reg_' + rn).replace(' ','_')
        #deltaB
        data2plot = df_summary_eq.loc[i_r,['eqlat','eqlon',cn_dBP]].values
        data2plot[:,2] /= scl_dBP
        #plot figure
        fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=-1.5, cmax=1.5, flag_grid=False, 
                                                              title=None, cbar_label='', log_cbar = False, 
                                                              frmt_clb = '%.2f', alpha_v = 0.7, cmap='seismic', 
                                                              marker_size=70.)
        #edit figure properties
        cbar.ax.tick_params(labelsize=28)
        cbar.set_label(r'$\delta BP$ (scaled)', size=30)
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
        plt.close(fig)
    
    #between-site residuals (spatial variability)
    for j, rid in enumerate(np.unique(df_summary_st.regid)):
        #region points
        i_r = df_summary_st.regid == rid
        #region name
        rn = df_summary_st.loc[i_r,'reg'].unique()[0]
        #figure name
        fname_fig = (fname_main_out + '_deltaS_reg_' + rn).replace(' ','_')
        #deltaB
        data2plot = df_summary_st.loc[i_r,['stlat','stlon',cn_dS]].values
        #plot figure
        fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=-3.0, cmax=3.0, flag_grid=False, 
                                                              title=None, cbar_label='', log_cbar = False, 
                                                              frmt_clb = '%.2f', alpha_v = 0.7, cmap='seismic', 
                                                              marker_size=70.)
        #edit figure properties
        cbar.ax.tick_params(labelsize=28)
        cbar.set_label(r'$\delta S$', size=30)
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
        plt.close(fig)
        
    #within-event-site residuals (spatial variability)
    for j, rid in enumerate(np.unique(df_summary_gm.regid)):
        #region points
        i_r = df_summary_gm.regid == rid
        #region name
        rn = df_summary_gm.loc[i_r,'reg'].unique()[0]
        #figure name
        fname_fig = (fname_main_out + '_deltaWS_reg_' + rn).replace(' ','_')
        #deltaB
        data2plot = df_summary_gm.loc[i_r,['stlat','stlon',cn_dWS]].values
        #plot figure
        fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=-4.0, cmax=4.0, flag_grid=False, 
                                                              title=None, cbar_label='', log_cbar = False, 
                                                              frmt_clb = '%.2f', alpha_v = 0.7, cmap='seismic', 
                                                              marker_size=70.)
        #edit figure properties
        cbar.ax.tick_params(labelsize=28)
        cbar.set_label(r'$\delta WS$', size=30)
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
        plt.close(fig)
        
    return None