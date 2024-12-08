#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:15:53 2024

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
import numpy.matlib as mlib
#statistics libraries
import pandas as pd
import geopandas as gpd
#projection libraries
import pyproj
#ground motion libraries
import pygmm
#user functions
sys.path.insert(0,'../../python_lib')
from pylib_gmm import scalingNL
from determine_style_of_faulting import W94sof
from pylib_clustering import find_indept_clusters

#%% Define Variables
### ======================================
#format version
flag_frmt_rev = 4
#censor option
flag_censor = True
#clustering option 
# flag_clusters = True
flag_clusters = False

#flatfiel filename
# fn_fltfile = '../../../Raw_files/flatfiles/NGAW3_FAS_Data_Version_1_20240704_OnlyMw.csv'
# fn_fltfile = '../../../Raw_files/flatfiles/NGAW3_FAS_Data_F_20240920_OnlyMwRmax.csv'
if flag_censor:
    # fn_fltfile        = '../../../Raw_files/flatfiles/NGAW3_FAS_Data_F_20240920_OnlyRmaxRegion.csv'
    # fn_fltfile          = '../../../Raw_files/flatfiles/flatfiles_20241116/NGAW3_FAS_Data_F_20241116_karen.csv'
    # fn_fltfile          = '../../../Raw_files/flatfiles/flatfiles_20241116/NGAW3_FAS_Data_F_20241116_karen_mod.csv'
    fn_fltfile          = '../../../Raw_files/flatfiles/flatfiles_20241205/NGAW3_FAS_Data_F_20241116_karen_v2.csv'
else:
    # fn_fltfile = '../../../Raw_files/flatfiles//NGAW3_FAS_Data_Version_0_20241116_All2.csv'
    # fn_fltfile = '../../../Raw_files/flatfiles/flatfiles_20241116/NGAW3_FAS_Data_Version_0_20241116_All2.csv'
    fn_fltfile = '../../../Raw_files/flatfiles/flatfiles_20241205/NGAW3_FAS_Data_Version_0_20241116_All2_v2.csv'

if flag_frmt_rev == 3:
    fn_fltfile_metadata = '../../../Raw_files/flatfiles/flatfiles_20241116/NGAW3_FAS_Data_Version_0_20241116_metadata.csv'

#file name region flatfile
fn_regions = '../../../Data/gis/regions/regions_global.shp'

#regionalization
reg  = {'PSW':['CA','Oregon','Mexico','New Mexico'], 
        'BR': ['Arizona','Colorado','Utah','Idaho','Wyoming','Nevada'],
        'PNW':['Alaska','Canada','Montana','Washington'], 
        'SEE':['Albania','Greece','Italy','Turkey','Macedonia','Montenegro'], 
        'CHN':['China'], 
        'JAP':['Japan'],  
        'TWN':['Taiwan'],  
        'NZ': ['New Zealand'],  
        'OTH':['Iran','Netherlands']}

#utm regions
reg_utm_zones = {'PSW':'11N',
                 'BR': '13N',
                 'PNW':'10N', 
                 'SEE':'34N', 
                 'CHN':'50N', 
                 'JAP':'54N',  
                 'TWN':'51N',  
                 'NZ': '59S',  
                 'OTH':'36N'}

#vs30 for non-linear effects
vsref_nl = 800.

#ground motion flatifle
if flag_censor:
    # fn_gm_flt = 'fltfile_nga3_20240920_censored'
    # fn_gm_flt = 'fltfile_nga3_20241116_censored'
    fn_gm_flt = 'fltfile_nga3_20241205_censored'
else:
    # fn_gm_flt = 'fltfile_nga3_20240920_all'
    fn_gm_flt = 'fltfile_nga3_20241116_all'
    fn_gm_flt = 'fltfile_nga3_20241205_all'

#output directory
dir_out = '../../../Data/gmm_ergodic/dataset/'

#%% Read 
### ======================================
#read flatfile
df_orig_flt = pd.read_csv(fn_fltfile, na_values=-999)
#read metadata
if flag_frmt_rev == 3:
    df_meta_flt = pd.read_csv(fn_fltfile_metadata)
    #collect closest point
    df_orig_flt = pd.merge(df_orig_flt, df_meta_flt[['motion_id','closest_point_latitude','closest_point_longitude','closest_point_depth']], 
                           left_on='motion_id', right_on='motion_id')

#rename new zeland issues
df_orig_flt.loc[df_orig_flt.region=='NewZealand','region'] = 'New Zealand'

#regions
if 'fn_regions' in locals():
  gdf_reg = gpd.read_file(fn_regions)

#%% Processing Variables
### ======================================
#identify unique events
_, eq_idx, eq_inv = np.unique(df_orig_flt[['event_id']].values, axis=0, return_inverse=True, return_index=True)
#create earthquake ids for all records (1 to n_eq)
eq_id = eq_inv + 1
n_eq = len(eq_idx)

#identify unique stations
_, st_idx, st_inv = np.unique(df_orig_flt[['station_id']].values, axis=0, return_inverse=True, return_index=True)
#create stationfor all records (1 to n_eq)
st_id = st_inv + 1
n_st = len(st_idx)

#number of ground motions
n_gm = len(df_orig_flt)

# Organize metadata
# -------------------------------
# gmm parameters
# ---   ---   ---   ---
motion_id = df_orig_flt['motion_id'].values
#event and station ids
event_id = df_orig_flt['event_id'].values
sta_id   = df_orig_flt['station_id'].values

#create cluster id
clust_id = np.zeros(n_gm)
if flag_clusters:
    clusters = find_indept_clusters(df_orig_flt[['event_id','station_id']].values)
    for j, clust in enumerate(clusters):
        clust_id[clust] = j+1

#event country
country_name = df_orig_flt['event_country'].values

#source parameters
mag    = df_orig_flt['magnitude'].values
ztor   = df_orig_flt['ztor'].values
strike = df_orig_flt['strike'].values
dip    = df_orig_flt['dip'].values
rake   = df_orig_flt['rake'].values
width  = df_orig_flt['fault_width'].values
area   = df_orig_flt['fault_area'].values

#set unspecified values to nan
mag[mag < -800]       = np.nan
ztor[ztor < -800]     = np.nan
strike[strike < -800] = np.nan
dip[dip < -800]       = np.nan
rake[rake < -800]     = np.nan
width[width < -800]   = np.nan
area[area < -800]     = np.nan

#determine sof
if not flag_censor:
    sof = np.array([W94sof(r) for r in rake])
else:
    sof = np.select([df_orig_flt.RN.values==1, df_orig_flt.RV.values==1], ['normal', 'reverse'], 'strike-slip')
#style of faulting value
sofid  = np.array([np.select([s.lower()=='reverse', s.lower()=='normal'], [1, -1], 0) 
                   for s in sof])
mech  = np.select([abs(sofid)<0.5, sofid>=0.5, sofid<=-0.5], ['SS','RS','NS'], 'None')
#mainshock/aftershock 
eqclass = np.ones(n_gm)
# eqclass = df_orig_flt['event_type_id'].values

#path parameters
if flag_frmt_rev  == 1:
    rrup = df_orig_flt['rrup'].values
    rjb	 = df_orig_flt['rjb'].values
    rx   = df_orig_flt['rx'].values
    ry   = df_orig_flt['ry'].values
    ry0  = df_orig_flt['ry0'].values
    ravg = df_orig_flt['ravg'].values
elif flag_frmt_rev == 2:
    rrup = df_orig_flt['rrup_m'].values
    rjb	 = df_orig_flt['Rjb_m'].values
    rx   = df_orig_flt['Rx_m'].values
    ry   = df_orig_flt['Ry_m'].values
    ry0  = df_orig_flt['Ry0_m'].values
    ravg = df_orig_flt['Ravg_m'].values
elif flag_frmt_rev == 3 or flag_frmt_rev == 4:
    rrup = df_orig_flt['rrup_m'].values
    rjb	 = df_orig_flt['rjb_m'].values
    rx   = df_orig_flt['rx_m'].values
    ry   = df_orig_flt['ry_m'].values
    ry0  = df_orig_flt['ry0_m'].values
    ravg = df_orig_flt['ravg_m'].values

#set unspecified values to nan
rrup[rrup < -800] = np.nan
ravg[ravg < -800] = np.nan
rjb[rjb < -800]   = np.nan
rx[rx < -800]     = np.nan
ry[ry < -800]     = np.nan
ry0[ry0 < -800]   = np.nan

#site
vs30       = df_orig_flt['vs30'].values
vs30_class = df_orig_flt['Vs30_class'].values
#set unspecified values to nan
vs30[vs30 < -800] = np.nan

#z1.0 and z2.5
if flag_frmt_rev == 1:
    z1p0_mat = df_orig_flt[['z1p0_measured','z1p0_CVMS4','z1p0_CVMS4.26','z1p0_CVMS4.26.M01','z1p0_CVMH15.1','z1p0_SFCVM21.1','z1p0_USGSNCM','z1p0_NIED']].values
    z2p5_mat = df_orig_flt[['z2p5_measured','z2p5_CVMS4','z2p5_CVMS4.26','z2p5_CVMS4.26.M01','z2p5_CVMH15.1','z2p5_SFCVM21.1','z2p5_USGSNCM','z2p5_NIED']].values
    #set unavailable entries to nan
    z1p0_mat[z1p0_mat==-999] = np.nan
    z2p5_mat[z2p5_mat==-999] = np.nan
    #initialize arrays
    z1p0 = np.full(n_gm, np.nan)
    z2p5 = np.full(n_gm, np.nan)
    for j in range(n_gm):
        #z1.0 and z2.5 indices
        i_z1p0_fnt = np.isfinite(z1p0_mat[j,:]) 
        i_z2p5_fnt = np.isfinite(z2p5_mat[j,:]) 
        #select values based on priority order
        z1p0[j] = z1p0_mat[j,np.argwhere(i_z1p0_fnt)[0][0]] if np.any(i_z1p0_fnt) else np.nan
        z2p5[j] = z2p5_mat[j,np.argwhere(i_z2p5_fnt)[0][0]] if np.any(i_z2p5_fnt) else np.nan
elif flag_frmt_rev >= 2:
    z1p0 = np.full(n_gm, np.nan)
    z2p5 = np.full(n_gm, np.nan)

# regionaization
# ---   ---   ---   ---
#intialize ids
reg_id = np.full(n_gm, -1)
#intialize names
reg_name = np.full(n_gm, 'TMP')
reg_utm  = np.full(n_gm, '00N')

#source regionalization
if 'gdf_reg' in locals():
    #geopanads event
    gdf_eq = gpd.GeoDataFrame(pd.DataFrame({'event_id':df_orig_flt.loc[eq_idx,'event_id'].values}),
                              geometry=gpd.points_from_xy(df_orig_flt.loc[eq_idx,'hypocenter_longitude'].values,
                                                          df_orig_flt.loc[eq_idx,'hypocenter_latitude'].values), 
                              crs=gdf_reg.crs)
    #add region information
    gdf_eq = gpd.sjoin(gdf_eq, gdf_reg, how="inner", op="within")
    #geopandas ground motoin 
    gdf_gm = pd.merge(df_orig_flt[['event_id']], gdf_eq, how='left')[['reg_id','reg_abrv','reg_utm']]

    #region id, name, and utm
    reg_id   = gdf_gm.reg_id.values
    reg_name = gdf_gm.reg_abrv.values
    reg_utm  = gdf_gm.reg_utm.values
    
    #region names
    _, reg_idx = np.unique(reg_id, return_index=True)
    reg = reg_name[reg_idx]

else:
    for j in range(n_gm):
        #iterate over regions
        for r_i, r_k in enumerate(reg, start=1):
            if np.isin(df_orig_flt.loc[j,'region'], reg[r_k]):
                #upate region gs
                reg_id[j]   = r_i
                reg_name[j] = r_k
    reg_utm[j] = reg_utm_zones[reg_name[j]]


#check full religonization
# assert(np.all(reg_id > 0)),'Error. Incomplete source regionalization.'
# df_orig_flt.loc[reg_id<0,'region'] #problematic source regionalization

# event/source location
# ---   ---   ---   ---
#event location
eq_lat    = df_orig_flt['hypocenter_latitude'].values
eq_lon    = df_orig_flt['hypocenter_longitude'].values
if   flag_frmt_rev <= 2:
    eq_z = df_orig_flt['hypocenter_depth'].values
elif flag_frmt_rev >= 3:
    eq_z = df_orig_flt['hypo_depth'].values

#event closest point
if   flag_frmt_rev <= 2:
    eqclt_lat = df_orig_flt['closest_point_latitude'].values
    eqclt_lon = df_orig_flt['closest_point_longitude'].values
    eqclt_z   = df_orig_flt['closest_point_depth'].values
else:
    eqclt_lat = df_orig_flt['closest_point_latitude'].values
    eqclt_lon = df_orig_flt['closest_point_longitude'].values
    eqclt_z   = df_orig_flt['closest_point_depth'].values
#station location
st_lat    = df_orig_flt['station_latitude'].values
st_lon    = df_orig_flt['station_longitude'].values

#set unspecified values to nan
eq_lat[eq_lat < -800]       = np.nan
eq_lon[eq_lon < -800]       = np.nan
eq_z[eq_z < -800]           = np.nan
eqclt_lat[eqclt_lat < -800] = np.nan
eqclt_lon[eqclt_lon < -800] = np.nan
eqclt_z[eqclt_z < -800]     = np.nan
st_lat[st_lat < -800]       = np.nan
st_lon[st_lon < -800]       = np.nan

#compute region specific coordinates
eq_xy    = np.full((n_gm,2), np.nan)
eqclt_xy = np.full((n_gm,2), np.nan)
st_xy    = np.full((n_gm,2), np.nan)

for j in range(n_gm):
    #utm zone and projection
    utm_zone = reg_utm[j]
    if type(utm_zone) is str:
        utm_proj = pyproj.Proj("+proj=utm +zone="+utm_zone[:-1]+" +ellps=WGS84 +datum=WGS84 +units=km +no_defs")
        #utm coordinates
        eq_xy[j,:]    = utm_proj(eq_lon[j], eq_lat[j])
        eqclt_xy[j,:] = utm_proj(eqclt_lon[j], eqclt_lat[j])
        st_xy[j,:]    = utm_proj(st_lon[j], st_lat[j])

# ground motions
# ---   ---   ---   ---
i_gm = np.array([bool(re.match('^eas.(.*)Hz', c)) for c in df_orig_flt.columns])
#frequency
freq = np.array([float(re.findall('^eas.(.*)Hz', c.replace('p','.'))[0]) for c in df_orig_flt.columns[i_gm]])
cn_eas_freq   = ['eas_f%.9fhz'%f for f in freq]
cn_easln_freq = ['easln_f%.9fhz'%f for f in freq]
#eas
eas = df_orig_flt.loc[:,i_gm].values
eas[eas == -999] = np.nan
eas[eas < 1e-13] = np.nan

# ground motions (without non-linear scaling)
# ---   ---   ---   ---
eas_med = np.zeros(eas.shape)
f_nl  = np.zeros(eas.shape)

#iterate over all scenarios
for j in range(n_gm):
    #ground motion scenario
    gmm_scen = pygmm.Scenario(mag=mag[j], dip=dip[j], mechanism=mech[j], 
                              width=width[j], depth_tor=ztor[j], 
                              dist_rup=rrup[j], dist_x=rx[j], dist_y0=ry0[j], dist_jb=rjb[j], 
                              v_s30=vsref_nl)
    
    #define eas gmm
    gmm_eas = pygmm.BaylessAbrahamson2019(gmm_scen)
    
    #compute median ground motions
    eas_med[j,:] = np.exp( np.interp( np.log(freq), np.log(gmm_eas.freqs),   np.log(gmm_eas.eas)) )

    #non-linear correction factor
    f_nl[j,:] = scalingNL(vs30[j], freq, eas_med[j,:])

#linearly corrected eas
eas_lin = eas / np.exp( f_nl )

# Summarize
# -------------------------------
#generate processed ground motion flatfile
df_gm_flt = pd.DataFrame({'motionid':motion_id, 
                          'regid':reg_id, 'eqid':eq_id, 'stid':st_id, 'clstid':clust_id, 
                          'eventid':event_id, 'stationid':sta_id})

#region info
df_gm_flt.loc[:,'reg']     = reg_name
#country
df_gm_flt.loc[:,'country'] = country_name

#eq lat/lon
df_gm_flt.loc[:,'eqlat'] = eq_lat
df_gm_flt.loc[:,'eqlon'] = eq_lon
#eq utm
df_gm_flt.loc[:,'eqx']   = eq_xy[:,0]
df_gm_flt.loc[:,'eqy']   = eq_xy[:,1]
df_gm_flt.loc[:,'eqz']   = eq_z

#closet point lat/lon
df_gm_flt.loc[:,'eqcltlat'] = eqclt_lat
df_gm_flt.loc[:,'eqcltlon'] = eqclt_lon
#closet point utm
df_gm_flt.loc[:,'eqcltx']   = eqclt_xy[:,0]
df_gm_flt.loc[:,'eqclty']   = eqclt_xy[:,1]
df_gm_flt.loc[:,'eqcltz']   = eqclt_z

#station lat/lon
df_gm_flt.loc[:,'stlat'] = st_lat
df_gm_flt.loc[:,'stlon'] = st_lon
#station utm
df_gm_flt.loc[:,'stx']   = st_xy[:,0]
df_gm_flt.loc[:,'sty']   = st_xy[:,1]

#source parameters
df_gm_flt.loc[:,'mag']       = mag
df_gm_flt.loc[:,'ztor']      = ztor
df_gm_flt.loc[:,'sof']       = sof
df_gm_flt.loc[:,'sofid']     = sofid
df_gm_flt.loc[:,'strike']    = strike
df_gm_flt.loc[:,'dip']       = dip
df_gm_flt.loc[:,'rake']      = rake
df_gm_flt.loc[:,'width']     = width
df_gm_flt.loc[:,'area']      = area
df_gm_flt.loc[:,'eqclass']   = eqclass 
#path parameters
df_gm_flt.loc[:,'rrup']      = rrup
df_gm_flt.loc[:,'rjb']       = rjb
df_gm_flt.loc[:,'rx']        = rx
df_gm_flt.loc[:,'ry']        = ry
df_gm_flt.loc[:,'ry0']       = ry0
df_gm_flt.loc[:,'ravg']      = ravg
#site parameters
df_gm_flt.loc[:,'vs30']      = vs30
df_gm_flt.loc[:,'vs30class'] = vs30_class
df_gm_flt.loc[:,'z1.0']      = z1p0
df_gm_flt.loc[:,'z2.5']      = z2p5
df_gm_flt.loc[:,'z1.0flag']  = z1p0
df_gm_flt.loc[:,'z2.5flag']  = z2p5

#ground motion
df_gm_flt.loc[:,cn_eas_freq]   = eas
#ground motion w/o non-linearity
df_gm_flt.loc[:,cn_easln_freq] = eas_lin

#remove ground motions on unavailable regions
i_reg = ~np.isnan(df_gm_flt.regid)
df_gm_flt       = df_gm_flt.loc[i_reg,:]
df_gm_flt.regid = df_gm_flt.regid.astype(int)

# %% Save data
# ======================================
# create output directories
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

#save processed dataframes
fn_gm_flt = '%s%s'%(dir_out, fn_gm_flt)
df_gm_flt.to_csv(fn_gm_flt+'.csv', index=False)

# %% Summary
# ======================================
#number of records per source region
msg = "Number of ground motions per source region\n"
for r_i, r_k in enumerate(reg, start=1):
    msg += "%s: %i\n"%(r_k, (df_gm_flt.reg==r_k).sum())
#number of events per source region
msg += "Number of events motions per source region\n"
for r_i, r_k in enumerate(reg, start=1):
    msg += "%s: %i\n"%(r_k, len(df_gm_flt.loc[df_gm_flt.reg==r_k,'eqid'].unique()))
#number of records per site region
msg += "Number of ground motions per site region\n"
for r_i, r_k in enumerate(reg, start=1):
    msg += "%s: %i\n"%(r_k, (df_gm_flt.reg==r_k).sum())
#number of sites per site region
msg += "Number of sites per site region\n"
for r_i, r_k in enumerate(reg, start=1):
    msg += "%s: %i\n"%(r_k, len(df_gm_flt.loc[df_gm_flt.reg==r_k,'stid'].unique()))
#number ground-motions removed
msg += "Removed ground motions\n"
msg += "GM w/o region: %i\n"%(np.sum(~i_reg))

#print summary
print(msg)
#write summary
f = open(dir_out + fn_gm_flt + '_summary' + '.txt', 'w')
f.write(msg)    
f.close()    

