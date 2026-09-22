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

data_hist_ssp245_MPI_ESM  = '/work/mh0033/m301036/LSAT/CMIP6-MPI-M-LR/MergeDataOut/tas_Amon_1850-2100_*.nc'

ds = xr.open_mfdataset(data_hist_ssp245_MPI_ESM, combine = 'nested', concat_dim = 'run')
ds

# In[3]:

#Perfrom land sea mask
land_sea_mask=xr.open_dataset('/work/mh0033/m301036/LSAT/CMIP6-MPI-M-LR/GR15_lsm_regrid.nc')

mask_data = land_sea_mask['var1']
mask_data
# Align the time coordinates between the mask dataset and the original dataset
time_values = pd.to_datetime(mask_data['time'].values, format='mixed', dayfirst=True)
mask_data['time'] = time_values

# Align the time coordinates between the mask dataset and the original dataset
mask_data = mask_data.reindex(time=ds['time'], method='nearest')

# Apply the land-sea mask to the original dataset
masked_tas = ds.where(mask_data == 1, drop=False)
masked_tas

# In[5]:
tas = masked_tas['tas'].loc[:,'1850-01-01':'2100-12-31',:,:]
tas = tas - 273.15
tas_climatology = tas.groupby('time.month').mean(dim='time')
tas_ano = tas.groupby('time.month') - tas_climatology
tas_ano
lat = tas_ano['lat']
lon = tas_ano['lon']

# In[6]:
#Separate data into monthly and output as new nc file
time_data = tas_ano['time']
time_index = pd.to_datetime(time_data.values)
time_index

# In[7]:
weights = np.cos(np.deg2rad(tas.lat))*xr.ones_like(tas['lon'])

# In[8]:
tas_ano_weighted = tas_ano.weighted(weights)
# display(tas_ano_weighted)
tas_ano_weighted_mean = tas_ano_weighted.mean(dim=['lat','lon'])
tas_ano_weighted_mean

# In[9]:
tas_ano_annual = tas_ano_weighted_mean.groupby('time.year').mean('time')
tas_ano_annual
# display(tas_ano_annual.min().values)
# tas_ano_annual.max().values

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
    
    season_months = tas_ano.sel(time=tas.time.dt.month.isin(months),lat=slice(0,90))
    
    # Calculate the seasonal mean SAT anomalies
    season_mean_anomalies = (season_months * weights).mean(dim=['lat', 'lon']) / weights.mean(dim=['lat', 'lon'])
    
    # Store the seasonal mean in the dictionary
    season_means[season] = season_mean_anomalies

# Access the multiyear JJA mean SAT anomalies
    
JJA_tas = season_means['JJA']
DJF_tas = season_means['DJF']
MAM_tas = season_means['MAM']
SON_tas = season_means['SON']

JJA_tas,DJF_tas,MAM_tas,SON_tas

# In[12]:
DJF_tas_mean = DJF_tas.groupby('time.year').mean(dim='time')
JJA_tas_mean = JJA_tas.groupby('time.year').mean(dim='time')
MAM_tas_mean = MAM_tas.groupby('time.year').mean(dim='time')
SON_tas_mean = SON_tas.groupby('time.year').mean(dim='time')

# JJA_tas_mean[0,:].values
DJF_tas_mean,JJA_tas_mean

# In[13]:
window_size =65
rolled = tas_ano_annual.rolling(year=65, center=True).construct("window_size")
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
tas_ano_annual= xr.DataArray(tas_ano_annual)
tas_ano_annual
# tas_ano_annual_reshaped = tas_ano_annual.broadcast_like(mask)
masked_annual_tas = mask*tas_ano_annual
masked_annual_tas

# In[17]:
#the data should be the masked_annual_tas
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
    trend[irun,:]=polyfit_run(masked_annual_tas.isel(run=irun).values)

# In[22]:
tas_ano_JJA= xr.DataArray(JJA_tas_mean)
tas_ano_JJA
# tas_ano_annual_reshaped = tas_ano_annual.broadcast_like(mask)
masked_JJA_tas = mask*tas_ano_JJA
masked_JJA_tas

# In[21]:
tas_ano_DJF= xr.DataArray(DJF_tas_mean)
tas_ano_DJF
# tas_ano_annual_reshaped = tas_ano_annual.broadcast_like(mask)
masked_DJF_tas = mask*tas_ano_DJF
masked_DJF_tas

tas_ano_MAM= xr.DataArray(MAM_tas_mean)
tas_ano_MAM
# tas_ano_annual_reshaped = tas_ano_annual.broadcast_like(mask)
masked_MAM_tas = mask*tas_ano_MAM
masked_MAM_tas

# In[19]:
tas_ano_SON= xr.DataArray(SON_tas_mean)
tas_ano_SON
# tas_ano_annual_reshaped = tas_ano_annual.broadcast_like(mask)
masked_SON_tas = mask*tas_ano_SON
masked_SON_tas

# In[23]:
trend_JJA = np.zeros((num_runs,num_windows))
for irun in range(30):
    trend_JJA[irun,:]=polyfit_run(masked_JJA_tas.isel(run=irun).values)
# In[24]:
trend_DJF = np.zeros((num_runs,num_windows))
for irun in range(30):
    trend_DJF[irun,:]=polyfit_run(masked_DJF_tas.isel(run=irun).values)
# In[25]:
trend_SON = np.zeros((num_runs,num_windows))
for irun in range(30):
    trend_SON[irun,:]=polyfit_run(masked_SON_tas.isel(run=irun).values)

# In[26]:
trend_MAM = np.zeros((num_runs,num_windows))
for irun in range(30):
    trend_MAM[irun,:]=polyfit_run(masked_MAM_tas.isel(run=irun).values)

# In[27]:
#Calculate the 15yr running trend time series
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import signal
from sklearn.linear_model import LinearRegression

# In[28]:
#output 30 run annual mean SAT
# tas_ano_annual_np = tas_ano_annual['tas'].values
x = np.arange(1850,2037,1)
num_time_series = tas_ano_annual.shape[0]

#calculate the ensemble mean of the MPI-ESM-LR 
tas_annual_mean = tas_ano_annual.mean('run')

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
CRUTEMP_annual = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/Data-output/CRUTEMP_annual_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
CRUTEMP_JJA = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/Data-output/CRUTEMP_JJA_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
CRUTEMP_DJF = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/Data-output/CRUTEMP_DJF_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
CRUTEMP_MAM = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/Data-output/CRUTEMP_MAM_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
CRUTEMP_SON = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/Data-output/CRUTEMP_SON_NH_65yr_trend.txt',delimiter='\t', skip_header=1)

CRUTEMP_trend_annual = CRUTEMP_annual[:,1]
CRUTEMP_trend_JJA = CRUTEMP_JJA[:,1]
CRUTEMP_trend_DJF = CRUTEMP_DJF[:,1]
CRUTEMP_trend_SON = CRUTEMP_SON[:,1]
CRUTEMP_trend_MAM = CRUTEMP_MAM[:,1]

# In[32]:

MLOST_annual = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/Data-output/MLOST_annual_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
MLOST_JJA = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/Data-output/MLOST_JJA_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
MLOST_DJF = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/Data-output/MLOST_DJF_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
MLOST_SON = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/Data-output/MLOST_SON_NH_65yr_trend.txt',delimiter='\t', skip_header=1)
MLOST_MAM = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/Data-output/MLOST_MAM_NH_65yr_trend.txt',delimiter='\t', skip_header=1)

MLOST_trend_annual = MLOST_annual[:,1]
MLOST_trend_JJA = MLOST_JJA[:,1]
MLOST_trend_DJF = MLOST_DJF[:,1]
MLOST_trend_MAM = MLOST_MAM[:,1]
MLOST_trend_SON = MLOST_SON[:,1]


# Extend MLOST_trend_annual to the same length as trend_mean
num_missing_years = len(trend_mean) - len(MLOST_trend_annual)
MLOST_trend_annual_extended = np.pad(MLOST_trend_annual, (0,num_missing_years), mode='constant', constant_values=np.nan)
CRUTEMP_trend_annual_extended = np.pad(CRUTEMP_trend_annual, (0,num_missing_years), mode='constant', constant_values=np.nan)

# In[33]:
MLOST_trend_DJF_extended = np.pad(MLOST_trend_DJF, (0,num_missing_years), mode='constant', constant_values=np.nan)
MLOST_trend_JJA_extended = np.pad(MLOST_trend_JJA, (0,num_missing_years), mode='constant', constant_values=np.nan)
MLOST_trend_MAM_extended = np.pad(MLOST_trend_MAM, (0,num_missing_years), mode='constant', constant_values=np.nan)
MLOST_trend_SON_extended = np.pad(MLOST_trend_SON, (0,num_missing_years), mode='constant', constant_values=np.nan)

CRUTEMP_trend_DJF_extended = np.pad(CRUTEMP_trend_DJF, (0,num_missing_years), mode='constant', constant_values=np.nan)
CRUTEMP_trend_JJA_extended = np.pad(CRUTEMP_trend_JJA, (0,num_missing_years), mode='constant', constant_values=np.nan)
CRUTEMP_trend_MAM_extended = np.pad(CRUTEMP_trend_MAM, (0,num_missing_years), mode='constant', constant_values=np.nan)
CRUTEMP_trend_SON_extended = np.pad(CRUTEMP_trend_SON, (0,num_missing_years), mode='constant', constant_values=np.nan)

# In[35]:

fig,axs = plt.subplots(5,1, figsize=(7,8.5), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0)

for i in range(num_time_series):
    axs[0].plot(x, trend[i, :]*65, color='gray')
    
axs[0].plot(x, CRUTEMP_trend_annual_extended*65, color='green', label='CRUTEM5')
axs[0].plot(x, MLOST_trend_annual_extended*65, color='blue', label='MLOST/Land')
axs[0].plot(x, trend_mean*65, color='black', label='MPI-ESM-LR-esm')
axs[0].set(ylim=(-2.5,3.0))
axs[0].set_title('NH LSATAs 65yrs running trend',fontsize=14)
axs[0].set_ylabel('Annual(°C/65yrs)', fontsize=14)
axs[0].set_xlabel('Start year', fontsize=14)
# axs[0].grid(visible=False, which='major', axis='y')
axs[0].tick_params(axis='x', labelsize=10)
axs[0].tick_params(axis='y', labelsize=10)
axs[0].legend()
# Plot the MAM time series
for i in range(num_time_series):
    axs[1].plot(x, trend_MAM[i, :]*65, color='gray')

axs[1].plot(x, CRUTEMP_trend_MAM_extended*65, color='green')
axs[1].plot(x, MLOST_trend_MAM_extended*65, color='blue')
axs[1].plot(x, trend_MAM_mean*65, color='black')
axs[1].set_ylabel('MAM(°C/65yrs)', fontsize=14)
axs[1].set_xlabel('Start year', fontsize=14)
axs[1].tick_params(axis='x', labelsize=10)
axs[1].tick_params(axis='y', labelsize=10)

# Plot the JJA time series
for i in range(num_time_series):
    axs[2].plot(x, trend_JJA[i, :]*65, color='gray')
    
axs[2].plot(x, CRUTEMP_trend_JJA_extended*65, color='green')
axs[2].plot(x, MLOST_trend_JJA_extended*65, color='blue')
axs[2].plot(x, trend_JJA_mean*65, color='black')
axs[2].set_ylabel('JJA(°C/65yrs)', fontsize=14)
axs[2].set_xlabel('Start year', fontsize=14)
# axs[2].grid(visible=False, which='major', axis='y')
axs[2].tick_params(axis='x', labelsize=10)
axs[2].tick_params(axis='y', labelsize=10)

# Plot the SON time series
for i in range(num_time_series):
    axs[3].plot(x, trend_SON[i, :]*65, color='gray')
    
axs[3].plot(x, CRUTEMP_trend_SON_extended*65, color='green')
axs[3].plot(x, MLOST_trend_SON_extended*65, color='blue')
axs[3].plot(x, trend_SON_mean*65, color='black')
axs[3].set_ylabel('SON(°C/65yrs)', fontsize=14)
axs[3].set_xlabel('Start year', fontsize=14)
axs[3].tick_params(axis='x', labelsize=10)
axs[3].tick_params(axis='y', labelsize=10)

# Plot the DJF time series
for i in range(num_time_series):
    axs[4].plot(x, trend_DJF[i, :]*65, color='gray')
    
axs[4].plot(x, CRUTEMP_trend_DJF_extended*65, color='green')
axs[4].plot(x, MLOST_trend_DJF_extended*65, color='blue')
axs[4].plot(x, trend_DJF_mean*65, color='black')

axs[4].set_ylabel('DJF(°C/65yrs)', fontsize=14)
axs[4].set_xlabel('Start year', fontsize=14)
axs[4].tick_params(axis='x', labelsize=10)
axs[4].tick_params(axis='y', labelsize=10)

plt.show()

# In[36]:
fig.savefig("./2100-MPI-ESM-LR-NH_SATAs_65yr_running_trend.png")

# In[37]:
# ## Extraction the 1958-2022 trend values
MLOST_DJF_trend = MLOST_trend_DJF[-1]*65.0
MLOST_DJF_trend = np.round(MLOST_DJF_trend, decimals=2)
print(MLOST_DJF_trend)

MLOST_JJA_trend = MLOST_trend_JJA[-1]*65.0
MLOST_JJA_trend = np.round(MLOST_JJA_trend, decimals=2)
print(MLOST_JJA_trend)

MLOST_SON_trend = MLOST_trend_SON[-1]*65.0
MLOST_SON_trend = np.round(MLOST_SON_trend, decimals=2)
print(MLOST_SON_trend)

MLOST_MAM_trend = MLOST_trend_MAM[-1]*65.0
MLOST_MAM_trend = np.round(MLOST_MAM_trend, decimals=2)
print(MLOST_MAM_trend)
# In[38]:
CRUTEMP_DJF_trend = CRUTEMP_trend_DJF[-1]*65.0
CRUTEMP_DJF_trend = np.round(CRUTEMP_DJF_trend, decimals=2)
print(CRUTEMP_DJF_trend)

CRUTEMP_JJA_trend = CRUTEMP_trend_JJA[-1]*65.0
CRUTEMP_JJA_trend = np.round(CRUTEMP_JJA_trend, decimals=2)
print(CRUTEMP_JJA_trend)

CRUTEMP_SON_trend = CRUTEMP_trend_SON[-1]*65.0
CRUTEMP_SON_trend = np.round(CRUTEMP_SON_trend, decimals=2)
print(CRUTEMP_SON_trend)

CRUTEMP_MAM_trend = CRUTEMP_trend_MAM[-1]*65.0
CRUTEMP_MAM_trend = np.round(CRUTEMP_MAM_trend, decimals=2)
print(CRUTEMP_MAM_trend)

# In[40]:
trend_2023_DJF = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_DJF[:,-14]*65.0})
trend_2023_DJF.to_csv('MPI-ESM-LR_NH_SATAs_2023_65yr_DJF_trend.txt', sep='\t', index=False)

trend_2023_JJA = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_JJA[:,-14]*65.0})
trend_2023_JJA.to_csv('MPI-ESM-LR_NH_SATAs_2023_65yr_JJA_trend.txt', sep='\t', index=False)

trend_2023_SON = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_SON[:,-14]*65.0})
trend_2023_SON.to_csv('MPI-ESM-LR_NH_SATAs_2023_65yr_SON_trend.txt', sep='\t', index=False)

trend_2023_MAM = pd.DataFrame({'run': np.arange(1,31,1), 'values': trend_MAM[:,-14]*65.0})
trend_2023_MAM.to_csv('MPI-ESM-LR_NH_SATAs_2023_65yr_MAM_trend.txt', sep='\t', index=False)


# %%
