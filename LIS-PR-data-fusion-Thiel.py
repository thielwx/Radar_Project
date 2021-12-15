#!/usr/bin/env python
# coding: utf-8

# In[6]:


# This program runs over one month of LIS and PR data, 
# and creates comparison files that will be merged into a bulk dataset.

# Author: Kevin Thiel
# Written: November 2021 for the METR 5673 Weather Radar and Theory Final Project


# In[7]:


#Reading in the necessary packages (see Merger-Playground for plotting capabilities)
import numpy as np
import netCDF4 as nc
import h5py
import pandas as pd
from datetime import datetime
import os


# In[8]:


#Taking the given file path and grabbing the lat and lons of the lis flashes
def lis_data_setup(file):
    dset = nc.Dataset(file)
    
    lis_lats = dset.variables['lightning_flash_location'][:,0]
    lis_lons = dset.variables['lightning_flash_location'][:,1]
    
    dset.close()
    
    return lis_lats, lis_lons


# In[69]:


#Taking the given file path and grabbing the lat, lon, precip rate near surface (prns), and overpass start time
def pr_data_setup(file):
    f = h5py.File(file, 'r')
    group = f['NS'] 
    
    pr_lat = group['Latitude'][:]
    pr_lon = group['Longitude'][:]
    slv = group['SLV']
    
    scantime = group['ScanTime']
    prns = slv['precipRateNearSurface'][:]
    
    #Selecting all of the non-zero data points, and making the zeros in nans
    prns[~(prns>0.0001)] = np.nan
    
    #Getting the time and making sure its all zero-padded to go into datetime
    year=scantime['Year'][:][0].astype('str')
    month=scantime['Month'][:][0]
    if month<10:
        month = '0'+month.astype('str')
    else:
        month = month.astype('str')

    day=scantime['DayOfMonth'][:][0]
    if day<10:
        day = '0'+day.astype('str')
    else:
        day = day.astype('str')

    hour=scantime['Hour'][:][0]
    if hour<10:
        hour = '0'+hour.astype('str')
    else:
        hour = hour.astype('str')

    minute=scantime['Minute'][:][0]
    if minute<10:
        minute = '0'+minute.astype('str')
    else:
        minute = minute.astype('str')

    cur_time_str = year+month+day+hour+minute

    cur_time = datetime.strptime(cur_time_str,'%Y%m%d%H%M')
    
    return pr_lat, pr_lon, prns, cur_time, cur_time_str


# In[10]:


#This function makes a target grid using our defined spacing
def grid_maker(dx):
    #Establishing the grid points in a 1-D and 2-D array
    lat_pts = np.arange(30,36.75,dx)
    lon_pts = np.arange(-120,-79.75,dx)
    lon_grid, lat_grid = np.meshgrid(lon_pts, lat_pts)
    
    #Will be used in the grid accumulation part later
    lat_index = np.arange(0,len(lat_pts))
    lon_index = np.arange(0,len(lon_pts))
    
    #Getting the target grids for LIS and PR data
    target_grid_lis = np.ones(lon_grid.shape)*np.nan
    target_grid_prns = np.ones(lon_grid.shape)*np.nan
    
    return lat_pts, lon_pts, lon_grid, lat_grid, lat_index, lon_index, target_grid_lis, target_grid_prns


# In[39]:


#Accumulating the LIS data to the target grid on a point-by-point basis
def ltg_grid_accum(lat_index, lon_index, lis_lats, lis_lons, dx, target_grid_lis, lat_pts, lon_pts):
    
    #Checking that there's any data to accumualte to the grid
    bounded_ltg_pts = np.where((lis_lats <= np.max(lat_pts))&
            (lis_lats >= np.min(lat_pts))&
            (lis_lons <= np.max(lon_pts))&
            (lis_lons >= np.min(lon_pts)))[0]
    
    #If there's ltg data in the bounds of the grid, then we'll accumulate to the grid
    if len(bounded_ltg_pts)>0:
        
        #Outer loop for looping through the defined latitudes
        for i in lat_index[:]:
            #Inner loop for looping through the defined longitudes
            for j in lon_index[:]:
                #Finding where the LIS points fall within the defined bounds
                lis_points = np.where((lis_lats < lat_pts[i]+(dx/2))&
                                      (lis_lats >= lat_pts[i]-(dx/2))&
                                      (lis_lons < lon_pts[j]+(dx/2))&
                                      (lis_lons >= lon_pts[j]-(dx/2)))[0]

                #If we find flashes in the searched domain, add the count to the target grid
                if len(lis_points)>0:
                    target_grid_lis[i,j] = len(lis_points)
        
    
    return target_grid_lis, bounded_ltg_pts


# In[67]:


#Accumulating the MAX precip rate near surface (prns) data to the target grid on a point-by-point basis
def pr_grid_accum(lat_index, lon_index, pr_lat, pr_lon, lat_pts, lon_pts, dx, target_grid_prns, prns):
    
    #Setting up all the variables that I want to capture
    prns_num_samples = target_grid_prns.copy()
    
    prns_max = target_grid_prns.copy()
    prns_90 = target_grid_prns.copy()
    prns_75 = target_grid_prns.copy()
    prns_mean = target_grid_prns.copy()
    prns_median = target_grid_prns.copy()
    prns_25 = target_grid_prns.copy()
    prns_10 = target_grid_prns.copy()
    prns_min = target_grid_prns.copy()
    
    #Flattening the data so it can be processed easier
    pr_lat_flat = pr_lat.flatten('C')
    pr_lon_flat = pr_lon.flatten('C')
    prns_flat = prns.flatten('C')

    #Accumulating the PR data to the target grid
    for i in lat_index[:]:
        for j in lon_index[:]:
            #Getting the points that exist in the gridbox from the overpass
            pr_points = np.where((pr_lat_flat < lat_pts[i]+(dx/2))&
                                  (pr_lat_flat >= lat_pts[i]-(dx/2))&
                                  (pr_lon_flat < lon_pts[j]+(dx/2))&
                                  (pr_lon_flat >= lon_pts[j]-(dx/2)))[0]
            
            #If there was an overpass within this gridbox
            if len(pr_points)>0:
                
                #Grabbing only the pr data that are relevant to the grid box
                prns_selected = prns_flat[pr_points]

                #Assigning the data to their appropriate grids
                prns_num_samples[i,j] = len(prns_selected)

                prns_max[i,j] = np.nanmax(prns_selected)
                prns_90[i,j] = np.nanpercentile(prns_selected, 90)
                prns_75[i,j] = np.nanpercentile(prns_selected, 75)
                prns_mean[i,j] = np.nanmean(prns_selected)
                prns_median[i,j] = np.nanpercentile(prns_selected, 50)
                prns_25[i,j] = np.nanpercentile(prns_selected, 25)
                prns_10[i,j] = np.nanpercentile(prns_selected, 10)
                prns_min[i,j] = np.nanmin(prns_selected)
                
            
    return prns_num_samples, prns_max, prns_90, prns_75, prns_mean, prns_median, prns_25, prns_10, prns_min


# In[66]:


#Taking all the grids and combining them into a dataframe file
def df_maker(prns_num_samples, prns_max, prns_90, prns_75, prns_mean, prns_median, prns_25, prns_10, prns_min, target_grid_lis, lon_grid, lat_grid, cur_time):
    
    #Flattening the data
    p_samples = prns_num_samples.flatten('C')
    p_max = prns_max.flatten('C')
    p_90 = prns_90.flatten('C')
    p_75 = prns_75.flatten('C')
    p_mean = prns_mean.flatten('C')
    p_median = prns_median.flatten('C')
    p_25 = prns_25.flatten('C')
    p_10 = prns_10.flatten('C')
    p_min = prns_min.flatten('C')
    
    lis = target_grid_lis.flatten('C')
    
    lon_grid_flat = lon_grid.flatten('C')
    lat_grid_flat = lat_grid.flatten('C')

    #Finding the indicies 
    combo_locations = np.where((p_max>0.)&(lis>0.))[0]

    #Applying the indicides with data to the flattened datasets
    psamples_final = p_samples[combo_locations]
    pmax_final = p_max[combo_locations]
    p90_final = p_90[combo_locations]
    p75_final = p_75[combo_locations]
    pmean_final = p_mean[combo_locations]
    pmedian_final = p_median[combo_locations]
    p25_final = p_25[combo_locations]
    p10_final = p_10[combo_locations]
    pmin_final = p_min[combo_locations]
    
    lis_final = lis[combo_locations]
    
    lon_final = lon_grid_flat[combo_locations]
    lat_final = lat_grid_flat[combo_locations]
    
    #Making a time list (start of TRMM overpass)
    time_final = np.broadcast_to([cur_time],len(combo_locations))
    
    #Creating the dataframe this can all go in (via dictionary)
    d = {'Time':time_final,
         'Lat':lat_final,
         'Lon':lon_final,
         'PrecipRateSAMPLES':psamples_final,
         'PrecipRateMAX':pmax_final,
         'PrecipRate90PERCENTILE':p90_final,
         'PrecipRate75PERCENTILE':p75_final,
         'PrecipRateMEAN':pmean_final,
         'PrecipRateMEDIAN':pmedian_final,
         'PrecipRate25PERCENTILE':p25_final,
         'PrecipRate10PERCENTILE':p10_final,
         'PrecipRateMIN':pmin_final,
         'LISFlashes':lis_final}
    df = pd.DataFrame(data=d)
    
    return df


# In[73]:


months = ['March','April','May']


#Outer-most loop that goes through month-by-month
for data_month in months:
    
    #Constants and file path locations
    dx = 0.5 

    drive_loc = '/Volumes/My Passport/Radar_Proj/'
    lis_data_loc = drive_loc+data_month+'-LIS/'
    pr_data_loc = drive_loc+data_month+'-PR/'
    output_loc = drive_loc+data_month+'-output-files-v2/'
    

    #Getting the list of files for the month
    lis_flist = sorted(os.listdir(lis_data_loc))
    pr_flist = sorted(os.listdir(pr_data_loc))

    findex = np.arange(0,len(lis_flist),1)

    #Loop that reads through each file (as long as the data matches)

    for i in findex:
        #Selecting the file name from the list
        lis_file = lis_flist[i]
        pr_file = pr_flist[i]

        #Checking that the files are matching
        lis_op_num = lis_file[-8:-3]
        pr_op_num = pr_file[-15:-10]

        if lis_op_num != pr_op_num:
            print ('---ERROR: FILE MISMATCH---')
            print ('LIS Overpass: '+lis_op_num)
            print ('PR  Overpass: '+pr_op_num)
            #quit()

        print ('--------------------------')
        print ('LIS Overpass: '+lis_op_num)
        print ('PR  Overpass: '+pr_op_num)

        #Reading in the data
        lis_lats, lis_lons = lis_data_setup(lis_data_loc+lis_file)
        pr_lat, pr_lon, prns, cur_time, cur_time_str = pr_data_setup(pr_data_loc+pr_file)
        
        print ('Time:         '+cur_time_str)

        #Setting up the target grid we'll accumualte our data on
        lat_pts, lon_pts, lon_grid, lat_grid, lat_index, lon_index, target_grid_lis, target_grid_prns = grid_maker(dx)
        
        #Accumulating our LIS flashes to the target grid
        target_grid_lis, bounded_ltg_pts = ltg_grid_accum(lat_index, lon_index, lis_lats, lis_lons, dx, target_grid_lis, lat_pts, lon_pts)
        
        #If there's lightning data that we put on the grid, then accumulate the PR data to the grid too
        if len(bounded_ltg_pts)>0:
            #Placing the local max rainfall rate on the target grid
            prns_num_samples, prns_max, prns_90, prns_75, prns_mean, prns_median, prns_25, prns_10, prns_min = pr_grid_accum(lat_index, lon_index, pr_lat, pr_lon, lat_pts, lon_pts, dx, target_grid_prns, prns)

            #Truning the target grids, lats, lons, and time into a dataframe
            df = df_maker(prns_num_samples, prns_max, prns_90, prns_75, prns_mean, prns_median, prns_25, prns_10, prns_min, target_grid_lis, lon_grid, lat_grid, cur_time)

            #Saving the file out (if it has data)
            if df.shape[0] > 0:
                df.to_pickle(output_loc+cur_time_str+'-'+lis_op_num+'.pkl')

                print ('Index: '+str(i)+'/'+str(findex[-1]))
                print ('Data points: '+str(df.shape[0]))

