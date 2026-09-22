#!/usr/bin/env python
# coding: utf-8

# In[1]:


import proplot as pplt
import numpy as np
import xarray as xr
import dask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob
import dask.array as da
from metpy.units import units
from scipy import stats
from scipy.stats import linregress

# In[2]:

data_hist_ssp245_MPI_ESM  = '/work/mh0033/m301036/LSAT/CMIP6-MPI-M-LR/MergeDataOut/psl_Amon_1850-2100_*.nc'

ds = xr.open_mfdataset(data_hist_ssp245_MPI_ESM, combine = 'nested', concat_dim = 'run')
ds

# In[5]:
slp = ds['psl'].loc[:,'1850-01-01':'2100-12-31',0:90,:]
slp = slp/100.0
slp_climatology = slp.groupby('time.month').mean(dim='time')
slp_ano = slp.groupby('time.month') - slp_climatology
slp_ano
lat = slp_ano['lat']
lon = slp_ano['lon']


# In[7]:
weights = np.cos(np.deg2rad(slp.lat))*xr.ones_like(slp['lon'])

# In[8]:
slp_ano_weighted = slp_ano.weighted(weights)
# display(slp_ano_weighted)
slp_ano_weighted_mean = slp_ano_weighted.mean(dim=['lat','lon'])
slp_ano_weighted_mean

# In[9]:
slp_ano_annual = slp_ano_weighted_mean.groupby('time.year').mean('time')
slp_ano_annual
# display(slp_ano_annual.min().values)
# slp_ano_annual.max().values

# In[11]:
seasons = ['JJA', 'DJF', 'MAM', 'SON']
season_means = {}

for season in seasons:
    if season == 'JJA':
        months = [6,7,8]
    elif season == 'DJF':
        months =[12,1,2]
    elif season == 'MAM':
        months = [3,4,5]
    elif season == 'SON':
        months = [9,10,11]
    
    season_months = slp_ano.sel(time=slp.time.dt.month.isin(months),lat=slice(0,90))
    
    # Calculate the seasonal mean SAT anomalies
    season_mean_anomalies = (season_months * weights).mean(dim=['lat', 'lon']) / weights.mean(dim=['lat', 'lon'])
    
    # Store the seasonal mean in the dictionary
    season_means[season] = season_mean_anomalies

# Access the multiyear JJA mean SAT anomalies
    
JJA_SLP = season_means['JJA']
DJF_SLP = season_means['DJF']
MAM_SLP = season_means['MAM']
SON_SLP = season_means['SON']

JJA_SLP,DJF_SLP,MAM_SLP,SON_SLP

# In[12]:
DJF_SLP_mean = DJF_SLP.groupby('time.year').mean(dim='time')
JJA_SLP_mean = JJA_SLP.groupby('time.year').mean(dim='time')
MAM_SLP_mean = MAM_SLP.groupby('time.year').mean(dim='time')
SON_SLP_mean = SON_SLP.groupby('time.year').mean(dim='time')

# JJA_SLP_mean[0,:].values
DJF_SLP_mean,JJA_SLP_mean

# In[13]:
window_size =65
rolled = slp_ano_annual.rolling(year=65, center=True).construct("window_size")
rolled[0,:,:]
# In[14]:
#mask 
nyears = 251
windows = nyears - window_size + 1
k=0
mask = np.zeros((windows, nyears))
for i in range(windows):
    mask[i,k:k+65]=1
    k=k+1
print(windows)
# In[15]:
mask[0],mask[1]

windows = xr.DataArray(np.arange(0,187,1), dims='windows')
year = xr.DataArray(np.arange(1850, 2101,1), dims='year')
new_dims = {'windows': windows, 'year': year}
mask = xr.DataArray(mask,dims=('windows','year'), coords=new_dims)
type(mask)

# In[16]:
slp_ano_annual= xr.DataArray(slp_ano_annual)
slp_ano_annual
# slp_ano_annual_reshaped = slp_ano_annual.broadcast_like(mask)
masked_annual_slp = mask*slp_ano_annual
masked_annual_slp

# In[17]:
#the data should be the masked_annual_SLP
#  dim is the selected run
#  trend is the returning running 15 year trend with the dimension size '101'
def polyfit_run(data):
    trend = np.zeros(187)
    for i in range(187):
        trend[i] = np.polyfit(range(65), data[i,:][data[i,:] != 0], deg=1)[0]
    return trend

# In[18]:
window_size = 65

num_runs= 30
num_points  = 251
num_windows = num_points - window_size + 1 

trend = np.zeros((num_runs,num_windows))

for irun in range(30):
    trend[irun,:]=polyfit_run(masked_annual_slp.isel(run=irun).values)

# In[19]:
SLP_ano_JJA= xr.DataArray(JJA_SLP_mean)
SLP_ano_JJA
# SLP_ano_annual_reshaped = SLP_ano_annual.broadcast_like(mask)
masked_JJA_SLP = mask*SLP_ano_JJA
masked_JJA_SLP

# In[20]:
SLP_ano_DJF= xr.DataArray(DJF_SLP_mean)
SLP_ano_DJF
# SLP_ano_annual_reshaped = SLP_ano_annual.broadcast_like(mask)
masked_DJF_SLP = mask*SLP_ano_DJF
masked_DJF_SLP

SLP_ano_MAM= xr.DataArray(MAM_SLP_mean)
SLP_ano_MAM
# SLP_ano_annual_reshaped = SLP_ano_annual.broadcast_like(mask)
masked_MAM_SLP = mask*SLP_ano_MAM
masked_MAM_SLP

# In[21]:
SLP_ano_SON= xr.DataArray(SON_SLP_mean)
SLP_ano_SON
# SLP_ano_annual_reshaped = SLP_ano_annual.broadcast_like(mask)
masked_SON_SLP = mask*SLP_ano_SON
masked_SON_SLP

# In[22]:
trend_JJA = np.zeros((num_runs,num_windows))
for irun in range(30):
    trend_JJA[irun,:]=polyfit_run(masked_JJA_SLP.isel(run=irun).values)
# In[24]:
trend_DJF = np.zeros((num_runs,num_windows))
for irun in range(30):
    trend_DJF[irun,:]=polyfit_run(masked_DJF_SLP.isel(run=irun).values)
# In[25]:
trend_SON = np.zeros((num_runs,num_windows))
for irun in range(30):
    trend_SON[irun,:]=polyfit_run(masked_SON_SLP.isel(run=irun).values)

# In[26]:
trend_MAM = np.zeros((num_runs,num_windows))
for irun in range(30):
    trend_MAM[irun,:]=polyfit_run(masked_MAM_SLP.isel(run=irun).values)

# In[27]:
#Calculate the 15yr running trend time series
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import signal
from sklearn.linear_model import LinearRegression

# In[28]:
#output 30 run annual mean SAT
# SLP_ano_annual_np = SLP_ano_annual['SLP'].values
x = np.arange(1850,2037,1)
num_time_series = slp_ano_annual.shape[0]

#calculate the ensemble mean of the MPI-ESM-LR 
slp_annual_mean = slp_ano_annual.mean('run')

# In[30]:
year = xr.DataArray(np.arange(1850,2037,1),dims='year')
print(len(year))
trend_mean = trend.mean((0))
trend_JJA_mean = trend_JJA.mean((0))
trend_DJF_mean = trend_DJF.mean((0))
trend_MAM_mean = trend_MAM.mean((0))
trend_SON_mean = trend_SON.mean((0))

trend_mean = xr.DataArray(trend_mean,dims=('year'), coords={'year': year})
trend_JJA_mean = xr.DataArray(trend_JJA_mean,dims=('year'), coords={'year': year})
trend_DJF_mean = xr.DataArray(trend_DJF_mean,dims=('year'), coords={'year': year})
trend_MAM_mean = xr.DataArray(trend_MAM_mean,dims=('year'), coords={'year': year})
trend_SON_mean = xr.DataArray(trend_SON_mean,dims=('year'), coords={'year': year})
trend_DJF_mean.shape

# In[31]:
#input trend data
NOAA_annual = np.genfromtxt(fname='/work/mh0033/m301036/Land_surf_temp/1850-2100Analyses/Data-output/SLP_annual_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
NOAA_JJA = np.genfromtxt(fname='/work/mh0033/m301036/Land_surf_temp/1850-2100Analyses/Data-output/SLP_JJA_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
NOAA_DJF = np.genfromtxt(fname='/work/mh0033/m301036/Land_surf_temp/1850-2100Analyses/Data-output/SLP_DJF_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
NOAA_MAM = np.genfromtxt(fname='/work/mh0033/m301036/Land_surf_temp/1850-2100Analyses/Data-output/SLP_MAM_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
NOAA_SON = np.genfromtxt(fname='/work/mh0033/m301036/Land_surf_temp/1850-2100Analyses/Data-output/SLP_SON_NH_65yr_trend.txt',delimiter='\t', skip_header=1)

NOAA_trend_annual = NOAA_annual[:,1]
NOAA_trend_JJA = NOAA_JJA[:,1]
NOAA_trend_DJF = NOAA_DJF[:,1]
NOAA_trend_SON = NOAA_SON[:,1]
NOAA_trend_MAM = NOAA_MAM[:,1]

# In[32]:
# Extend MLOST_trend_annual to the same length as trend_mean
num_missing_years = len(trend_mean) - len(NOAA_trend_annual)
NOAA_trend_annual_extended = np.pad(NOAA_trend_annual, (0,num_missing_years), mode='constant', constant_values=np.nan)

# In[33]:
NOAA_trend_DJF_extended = np.pad(NOAA_trend_DJF, (0,num_missing_years), mode='constant', constant_values=np.nan)
NOAA_trend_JJA_extended = np.pad(NOAA_trend_JJA, (0,num_missing_years), mode='constant', constant_values=np.nan)
NOAA_trend_MAM_extended = np.pad(NOAA_trend_MAM, (0,num_missing_years), mode='constant', constant_values=np.nan)
NOAA_trend_SON_extended = np.pad(NOAA_trend_SON, (0,num_missing_years), mode='constant', constant_values=np.nan)

# In[35]:
fig,axs = plt.subplots(5,1, figsize=(7,8.5), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0)

for i in range(num_time_series):
    axs[0].plot(x, trend[i, :]*65, color='gray')
    
axs[0].plot(x, NOAA_trend_annual_extended, color='green', label='20Century')
axs[0].plot(x, trend_mean*65, color='black', label='MPI-ESM-LR-esm')
axs[0].set(ylim=(-1.0,1.0))
axs[0].set_title('NH SLP 65yrs running trend',fontsize=14)
axs[0].set_ylabel('Annual(hPa/65yrs)', fontsize=14)
axs[0].set_xlabel('Start year', fontsize=14)
# axs[0].grid(visible=False, which='major', axis='y')
axs[0].tick_params(axis='x', labelsize=10)
axs[0].tick_params(axis='y', labelsize=10)
axs[0].legend()
# Plot the MAM time series
for i in range(num_time_series):
    axs[1].plot(x, trend_MAM[i, :]*65, color='gray')

axs[1].plot(x, NOAA_trend_MAM_extended, color='green')
axs[1].plot(x, trend_MAM_mean*65, color='black')
axs[1].set_ylabel('MAM', fontsize=14)
axs[1].set_xlabel('Start year', fontsize=14)
axs[1].tick_params(axis='x', labelsize=10)
axs[1].tick_params(axis='y', labelsize=10)

# Plot the JJA time series
for i in range(num_time_series):
    axs[2].plot(x, trend_JJA[i, :]*65, color='gray')
    
axs[2].plot(x, NOAA_trend_JJA_extended, color='green')
axs[2].plot(x, trend_JJA_mean*65, color='black')
axs[2].set_ylabel('JJA', fontsize=14)
axs[2].set_xlabel('Start year', fontsize=14)
# axs[2].grid(visible=False, which='major', axis='y')
axs[2].tick_params(axis='x', labelsize=10)
axs[2].tick_params(axis='y', labelsize=10)

# Plot the SON time series
for i in range(num_time_series):
    axs[3].plot(x, trend_SON[i, :]*65, color='gray')
    
axs[3].plot(x, NOAA_trend_SON_extended, color='green')
axs[3].plot(x, trend_SON_mean*65, color='black')
axs[3].set_ylabel('SON', fontsize=14)
axs[3].set_xlabel('Start year', fontsize=14)
axs[3].tick_params(axis='x', labelsize=10)
axs[3].tick_params(axis='y', labelsize=10)

# Plot the DJF time series
for i in range(num_time_series):
    axs[4].plot(x, trend_DJF[i, :]*65, color='gray')
    
axs[4].plot(x, NOAA_trend_DJF_extended, color='green')
axs[4].plot(x, trend_DJF_mean*65, color='black')

axs[4].set_ylabel('DJF(hPa/65yrs)', fontsize=14)
axs[4].set_xlabel('Start year', fontsize=14)
axs[4].tick_params(axis='x', labelsize=10)
axs[4].tick_params(axis='y', labelsize=10)

plt.show()

# In[36]:
fig.savefig("./2100-MPI-ESM-LR-NH_SLPAs_65yr_running_trend.png")

# In[38]:
ERA5_annual = np.genfromtxt(fname='/work/mh0033/m301036/Land_surf_temp/1850-2100Analyses/Data-output/SLP_annual_NH_1900-2022_65yr_trend.txt',delimiter='\t', skip_header=1)
ERA5_JJA = np.genfromtxt(fname='/work/mh0033/m301036/Land_surf_temp/1850-2100Analyses/Data-output/SLP_JJA_NH_1900-2022_65yr_trend.txt',delimiter='\t', skip_header=1)
ERA5_DJF = np.genfromtxt(fname='/work/mh0033/m301036/Land_surf_temp/1850-2100Analyses/Data-output/SLP_DJF_NH_1900-2022_65yr_trend.txt',delimiter='\t', skip_header=1)
ERA5_MAM = np.genfromtxt(fname='/work/mh0033/m301036/Land_surf_temp/1850-2100Analyses/Data-output/SLP_MAM_NH_1900-2022_65yr_trend.txt',delimiter='\t', skip_header=1)
ERA5_SON = np.genfromtxt(fname='/work/mh0033/m301036/Land_surf_temp/1850-2100Analyses/Data-output/SLP_SON_NH_1900-2022_65yr_trend.txt',delimiter='\t', skip_header=1)

ERA5_trend_annual = ERA5_annual[:,1]
ERA5_trend_JJA = ERA5_JJA[:,1]
ERA5_trend_DJF = ERA5_DJF[:,1]
ERA5_trend_SON = ERA5_SON[:,1]
ERA5_trend_MAM = ERA5_MAM[:,1]
ERA5_DJF_trend = ERA5_trend_DJF[-1]
ERA5_DJF_trend = np.round(ERA5_DJF_trend, decimals=2)
print(ERA5_DJF_trend)

ERA5_JJA_trend = ERA5_trend_JJA[-1]
ERA5_JJA_trend = np.round(ERA5_JJA_trend, decimals=2)
print(ERA5_JJA_trend)

ERA5_SON_trend = ERA5_trend_SON[-1]
ERA5_SON_trend = np.round(ERA5_SON_trend, decimals=2)
print(ERA5_SON_trend)

ERA5_MAM_trend = ERA5_trend_MAM[-1]
ERA5_MAM_trend = np.round(ERA5_MAM_trend, decimals=2)
print(ERA5_MAM_trend)

# In[40]:
trend_2023_DJF = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_DJF[:,-14]*65.0})
trend_2023_DJF.to_csv('MPI-ESM-LR_NH_SLPAs_2023_65yr_DJF_trend.txt', sep='\t', index=False)

trend_2023_JJA = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_JJA[:,-14]*65.0})
trend_2023_JJA.to_csv('MPI-ESM-LR_NH_SLPAs_2023_65yr_JJA_trend.txt', sep='\t', index=False)

trend_2023_SON = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_SON[:,-14]*65.0})
trend_2023_SON.to_csv('MPI-ESM-LR_NH_SLPAs_2023_65yr_SON_trend.txt', sep='\t', index=False)

trend_2023_MAM = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_MAM[:,-14]*65.0})
trend_2023_MAM.to_csv('MPI-ESM-LR_NH_SLPAs_2023_65yr_MAM_trend.txt', sep='\t', index=False)


# %%
trend_1958_DJF = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_DJF[:,-14]*65.0})
trend_1958_DJF.to_csv('MPI-ESM-LR_NH_SLPAs_1958_65yr_DJF_trend.txt', sep='\t', index=False)

trend_1958_JJA = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_JJA[:,-14]*65.0})
trend_1958_JJA.to_csv('MPI-ESM-LR_NH_SLPAs_1958_65yr_JJA_trend.txt', sep='\t', index=False)

trend_1958_SON = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_SON[:,-14]*65.0})
trend_1958_SON.to_csv('MPI-ESM-LR_NH_SLPAs_1958_65yr_SON_trend.txt', sep='\t', index=False)

trend_1958_MAM = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_MAM[:,-14]*65.0})
trend_1958_MAM.to_csv('MPI-ESM-LR_NH_SLPAs_1958_65yr_MAM_trend.txt', sep='\t', index=False)
