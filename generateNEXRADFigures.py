import boto3
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pyproj import Geod
import math

##Create colorbar for instanenous rainfall rates
colors = [(1,1,1),(219/255,224/255,251/255), (10/255,80/255,204/255),(36/255,149/255,67/255),
          (166/255,48/255,167/255),(204/255,177/255,2/255),(204/255,122/255,1/255),
          (204/255,65/255,0/255),(153/255,6/255,1/255),(76/255,0/255,0/255)]  # R -> G -> B
n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
cmap_rain = LinearSegmentedColormap.from_list(cmap_name, colors, N=11)
bounds = [0,0.25,25.,50,75,100,130,150,175,200,250,500]
rain_norm = mpl.colors.BoundaryNorm(bounds, cmap_rain.N)

##Connect to AWS and retrieve data file
s3 = boto3.resource('s3', config=Config(signature_version=botocore.UNSIGNED,
                                        user_agent_extra='Resource'))
bucket = s3.Bucket('noaa-nexrad-level2')
for obj in bucket.objects.filter(Prefix='2021/03/25/KMXX/KMXX20210325_223801_V06'):
    f = Level2File(obj.get()['Body'])

##Pull required variables from data file
sweep = 0
az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])

ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

rLAT = f.sweeps[0][0][1].lat
rLON = f.sweeps[0][0][1].lon

##Create cmap for refelectivity
ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

indx = 0
var_data = ref
var_range = ref_range
colors = ref_cmap
lbl = 'REF (dBZ)'

data = np.ma.array(var_data)
data[np.isnan(data)] = np.ma.masked

##Cmap normalization.
norm = ref_norm if colors == ref_cmap else None

##Creates the base map.
def new_map(fig):
    proj = ccrs.PlateCarree()
    
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.coastlines('50m', 'black', linewidth=2, zorder=2)

    state_borders = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_1_states_provinces_lines',
        scale='50m', facecolor='none')
    ax.add_feature(state_borders, edgecolor='black', linewidth=1, zorder=3)
    
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    return ax

##Convert Z to linear units and then acquire rainfall rate 
var_dataMAG = np.power(10,var_data/10)
Rain_R = np.power(var_dataMAG/300,1/1.4)

##Create array to temporarily hold rainfall rates
temp_Rain_R = np.full((Rain_R.shape),np.nan)
temp_Rain_R[:] = Rain_R

##Create array for average rainfall rates and fill with nan
avg_Rain_R = np.full((Rain_R.shape),np.nan)
avg_Rain_R[:] = np.nan

##Convert x and y coordinates to lat and lon
g = Geod(ellps='clrk66')
rng = np.linspace(0, var_range[-1], data.shape[-1] + 1)
cen_lat = np.ones([len(az),len(rng)])*rLAT    
cen_lon = np.ones([len(az),len(rng)])*rLON
az2D = np.ones_like(cen_lat)*az[:,None]
rng2D = np.ones_like(cen_lat)*np.transpose(rng[:,None])*1000
lon,lat,back=g.fwd(cen_lon,cen_lat,az2D,rng2D)

pc_proj = ccrs.PlateCarree()

##For some weird reason the array is one column too large(don't ask me why I don't know). This fixes that.
lon = np.delete(lon,0,1)
lat = np.delete(lat,0,1)

## Parameters for generating the gridded average rainfall rates
lonMinRange = -88.5
numOfLonBox = 7
lonIter = 0.0
latMinRange = 32.0
numOfLatBox = 6
latIter = 0.0


##Generate reflectivity plot
fig = plt.figure(figsize=(15, 15))
ax = new_map(fig)
ax.set_title("Reflectivity")
ax.set_extent([-88.5, -85, 32, 35])
ax.add_feature(cfeature.STATES)
cm = ax.pcolormesh(lon, lat, var_data,norm=norm,cmap=ref_cmap,zorder=1,transform=pc_proj)
plt.colorbar(cm,orientation="vertical",fraction=0.040, pad=0.02,label = "Legend: dBZ")
plt.show()

##Generate instanenous rainfall rate plot
fig = plt.figure(figsize=(15, 15))
ax = new_map(fig)
ax.set_title("Nexrad derived rainfall rates")
ax.set_extent([-88.5, -85, 32, 35])
ax.add_feature(cfeature.STATES)
cm = ax.pcolormesh(lon, lat, Rain_R,norm=rain_norm,cmap=cmap_rain,zorder=1,transform=pc_proj)
plt.colorbar(cm,orientation="vertical",fraction=0.040, pad=0.02,label = "Legend: mm/hr")
plt.show()

##Generate average rainfall rate array
for latBox in range(numOfLatBox):
    for lonBox in range(numOfLonBox):
        for i in range(lon.shape[0]):
            for j in range(lon.shape[1]):
                if (lon[i][j] > lonMinRange+0.5+lonIter or lon[i][j] < lonMinRange+lonIter):
                    temp_Rain_R[i][j] = np.nan

        for i in range(lat.shape[0]):
            for j in range(lat.shape[1]):
                if (lat[i][j] > latMinRange+0.5+latIter or lat[i][j] < latMinRange+latIter):
                    temp_Rain_R[i][j] = np.nan

        avgVal = np.nanmean(temp_Rain_R)
        for i in range(lon.shape[0]):
            for j in range(lon.shape[1]):
                if lon[i][j] < lonMinRange+0.5+lonIter and lon[i][j] > lonMinRange+lonIter:
                    if lat[i][j] < latMinRange+0.5+latIter and lat[i][j] > latMinRange+latIter:
                        avg_Rain_R[i][j] = avgVal
        lonIter += 0.5
        temp_Rain_R[:] = Rain_R
    lonIter = 0.0
    latIter += 0.5

##Generate average rainfall rate plot            
fig = plt.figure(figsize=(15, 15))
ax = new_map(fig)
ax.set_title("Average rainfall rates")
ax.set_extent([-88.5, -85, 32, 35])
ax.add_feature(cfeature.STATES)
cm = ax.pcolormesh(lon, lat, avg_Rain_R,zorder=1,transform=pc_proj)
plt.colorbar(cm,orientation="vertical",fraction=0.040, pad=0.02,label = "Legend: mm/hr")
plt.show()

