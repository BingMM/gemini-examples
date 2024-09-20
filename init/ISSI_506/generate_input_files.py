#%% Import

import shutil
import sys
import numpy as np
import h5py
import gemini3d.read
import copy
from tqdm import tqdm
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

base = '/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/repos/ISSI_506'

sys.path.append(base)
import helpers_init

#%% Read simulation grid file

xg = gemini3d.read.grid(base + '/data/inputs/simgrid.h5')

#%% Get relevant grid information

mlat_sim    = 90 - xg['theta']/np.pi*180
mlon_sim    = xg['phi']/np.pi*180
r_sim       = xg['r']/1e3
Bmag_sim    = xg['Bmag']

# We need a 512x512 grid for input. This is 512x128. Luckily everthing is symmetric
mlat    = np.tile(mlat_sim, (1,1,4))
r       = np.tile(r_sim,    (1,1,4))
Bmag    = np.tile(Bmag_sim, (1,1,4))
mlon    = np.tile(np.linspace(np.min(mlon_sim), np.max(mlon_sim), 512), (512, 512, 1))

#%% Set hyperparameters

RE      = 6371.2
h_ref   = 110 # Reference height for FAC calculation

#%% Calculate upward continuation ratios and mlat/mlon from h_ref slice

up_ratio = np.zeros((Bmag.shape[1], Bmag.shape[2]))
mlat_fac = np.zeros(up_ratio.shape)
mlon_fac = np.zeros(up_ratio.shape)

for i in range(Bmag.shape[1]): # Loop over mlats (kinda, actually x2)
    for j in range(Bmag.shape[2]): # Loop over mlons (x3)
        k = np.argmin(abs(r[:, i, j] - RE - h_ref)) # id closest to h_ref in each column (x1)
        up_ratio[i, j] = Bmag[0, i, j] / Bmag[k, i, j]
        mlat_fac[i, j] = mlat[k, i, j]
        mlon_fac[i, j] = mlon[k, i, j]

#%% Determine fac input parameters

t_step_size = 10 # 10 # in seconds
sigmat      = 15 # t should actually be determine from this...
nsigma      = 3
duration    = sigmat*nsigma*2 # Duration of simulation in minutes
ts          = np.arange(0, duration*60+t_step_size, t_step_size)/60
centerlat   = 69 # Center of the FAC structure
mlat_tweak  = -2 # The equatorward tail is heavier than the poleward, I don't know why...
centerlon   = mlon_fac.mean() # Center magnetic longitude
width       = 90 # Longitudinal width

#%% Calculate FAC pattern at reference height

fac = np.zeros((mlon_fac.shape[0], mlon_fac.shape[1], len(ts)))
for i, t in tqdm(enumerate(ts), total=len(ts)):
    fac[:, :, i] = helpers_init.fac_input(t, mlon_fac, mlat_fac, 
                                          duration=duration, sigmat=sigmat,
                                          centerlat=centerlat, mlat_tweak=mlat_tweak, 
                                          centerlon=centerlon, width=width)

#%% Determine energy flux input parameters

Qpeak       = 25
Qbackground = .5
Qoval       = 5

#%% Calculate energy flux

fac_scaled = fac / np.max(np.abs(fac))

Q = (-1*fac_scaled) * Qpeak
Q[Q < Qbackground] = Qbackground
for i in range(Q.shape[2]):
    Q[:, :, i] += Qoval * np.exp(-(mlat_fac-centerlat)**2 / 2 / 4**2) # lazy

#%% Upward continuation

mlat_top = mlat[0, :, :]
mlon_top = mlon[0, :, :]

fac_top = np.zeros(fac.shape)
Q_top   = np.zeros(fac.shape)
for i in range(fac.shape[2]): # Lazy, again
    fac_top[:, :, i] = fac[:, :, i] * up_ratio
    Q_top[:, :, i]   = Q[:, :, i] * up_ratio

#%% Define Gaussian drop-off

# id_start = 100 # hack version 1-4
id_start = 75 # hack version 5

nstd = 5
std = id_start/nstd
x = np.arange(id_start+1)
y = 1/np.sqrt(2*np.pi*std**2)*np.exp(-(x-id_start)**2/(2*std**2))
y /= np.max(y)

# mask mlon
mask_mlon = np.ones((512, 512)) # Left right

mask_side = np.tile(y, (512, 1))

mask_mlon[:, :id_start+1] = mask_side
mask_mlon[:, -id_start-1:] = np.flip(mask_side, axis=1)

# mask mlat
mask_mlat = np.ones((512, 512)) # Top bot

#mask_mlat[:mlat_bot_id, :] = 0
#mask_mlat[mlat_bot_id:mlat_bot_id+id_start+1, :] = mask_side.T
mask_mlat[:id_start+1, :] = mask_side.T

#mask_mlat[mlat_top_id:, :] = 0
#mask_mlat[mlat_top_id-id_start-1:mlat_top_id, :] = np.flip(mask_side.T, axis=0)
mask_mlat[-id_start-1:, :] = np.flip(mask_side.T, axis=0)

# Create mask
f = mask_mlon < mask_mlat
mask = copy.deepcopy(mask_mlat)
mask[f] = mask_mlon[f]

#%% Interpolation unto input grids

with h5py.File(base + '/data/inputs/placeholders/simgrid.h5', 'r') as f:
    mlat_BC = f['mlat'][...]
    mlon_BC = f['mlon'][...]

mlat_min = np.min(90 - xg['theta']/np.pi*180)
mlat_max = np.max(90 - xg['theta']/np.pi*180)
mlat_buff = abs(mlat_max - np.max(mlat_BC))
mlat_BC = np.linspace(mlat_min-mlat_buff, mlat_max+mlat_buff, mlat_BC.size)

#mlat_BC, mlon_BC = np.meshgrid(mlat_BC, mlon_BC, indexing='ij')

fac_BC = np.zeros(fac_top.shape)
Q_BC = np.zeros(fac_top.shape)

for i in tqdm(range(len(ts)), total=len(ts)):
    
    f_fac = interp2d(mlon[0, 0, :], mlat[0, :, 0], fac_top[:, :, i], kind='linear', bounds_error=False, fill_value=None)
    f_Q   = interp2d(mlon[0, 0, :], mlat[0, :, 0], Q_top[:, :, i],   kind='linear', bounds_error=False, fill_value=None)

    fac_BC[:, :, i] = f_fac(mlon_BC, mlat_BC)
    Q_BC[:, :, i]   = f_Q(mlon_BC, mlat_BC)

#%% Make fields and precip input

t_start = 3600

for i in tqdm(range(len(ts)), total=len(ts)):
    t = t_start + i*t_step_size

    # Fields
    filename = base + '/data/inputs/fields/20160303_{}.000000.h5'.format(str(t).zfill(5))
    shutil.copy(base + '/data/inputs/placeholders/fields_20160303_15300.000000.h5', filename)

    with h5py.File(filename, 'r+') as f:
        f['Vminx1it'][...] = fac_top[:, :, i].astype(np.float32)
        f['time']['UTsec'][...] = float(t)

    # Precip
    filename = base + '/data/inputs/precip/20160303_{}.000000.h5'.format(str(t).zfill(5))
    shutil.copy(base + '/data/inputs/placeholders/precip_20160303_15300.000000.h5', filename)

    with h5py.File(filename, 'r+') as f:
        f['Qp'][...] = Q_top[:, :, i].astype(np.float32)
        f['time']['UTsec'][...] = float(t)

# Grids
shutil.copy(base + '/data/inputs/placeholders/simgrid.h5', 
            base + '/data/inputs/fields/simgrid.h5')
shutil.copy(base + '/data/inputs/placeholders/simgrid.h5', 
            base + '/data/inputs/precip/simgrid.h5')

shutil.copy(base + '/data/inputs/placeholders/simsize.h5', 
            base + '/data/inputs/fields/simsize.h5')
shutil.copy(base + '/data/inputs/placeholders/simsize.h5', 
            base + '/data/inputs/precip/simsize.h5')

with h5py.File(base + '/data/inputs/fields/simgrid.h5', 'r+') as f:
    f['mlat'][...] = mlat_BC.astype(np.float32)

with h5py.File(base + '/data/inputs/precip/simgrid.h5', 'r+') as f:
    f['mlat'][...] = mlat_BC.astype(np.float32)

# Initial conditions
shutil.copy(base + '/data/inputs/placeholders/initial_conditions.h5', 
            base + '/data/inputs/initial_conditions.h5')

with h5py.File(base + '/data/inputs/initial_conditions.h5', 'r+') as f:
    f['time']['UTsec'][...] = float(t_start)

#%% Plot BCs
'''
t_id = 220

plt.ioff()

mlat_grid, mlon_grid = np.meshgrid(mlat_BC, mlon_BC, indexing='ij')

fig, axs = plt.subplots(3, 2, figsize=(15, 20))
vmax = np.max(abs(fac))
axs[0,0].tricontourf(mlon_fac.flatten(), mlat_fac.flatten(), fac[:, :, t_id].flatten(), 
                   cmap='bwr', levels=np.linspace(-vmax, vmax, 40))
axs[1,0].tricontourf(mlon_top.flatten(), mlat_top.flatten(), fac_top[:, :, t_id].flatten(), 
                   cmap='bwr', levels=np.linspace(-vmax, vmax, 40))
axs[2,0].tricontourf(mlon_grid.flatten(), mlat_grid.flatten(), fac_BC[:, :, t_id].flatten(), 
                   cmap='bwr', levels=np.linspace(-vmax, vmax, 40))

vmax = np.max(Q)
axs[0,1].tricontourf(mlon_fac.flatten(), mlat_fac.flatten(), Q[:, :, t_id].flatten(), 
                   cmap='magma', levels=np.linspace(0, vmax, 40))
axs[1,1].tricontourf(mlon_top.flatten(), mlat_top.flatten(), Q_top[:, :, t_id].flatten(), 
                   cmap='magma', levels=np.linspace(0, vmax, 40))
axs[2,1].tricontourf(mlon_grid.flatten(), mlat_grid.flatten(), Q_BC[:, :, t_id].flatten(), 
                   cmap='magma', levels=np.linspace(0, vmax, 40))

for ax in axs[:, 0]:
    ax.set_ylabel('mlat')

for ax in axs[-1, :]:
    ax.set_xlabel('mlon')

for ax in axs.flatten():
    ax.grid()
    ax.set_ylim([47, 85])

axs[0,0].text(.5, 1.05, 'FAC at reference height (110)', ha='center', transform=axs[0,0].transAxes)
axs[0,1].text(.5, 1.05, 'Q  at reference height (110)', ha='center', transform=axs[0,1].transAxes)
axs[1,0].text(.5, 1.05, 'FAC at top of sim grid', ha='center', transform=axs[1,0].transAxes)
axs[1,1].text(.5, 1.05, 'Q at top of sim grid', ha='center', transform=axs[1,1].transAxes)
axs[2,0].text(.5, 1.05, 'FAC interpolated unto BC grid', ha='center', transform=axs[2,0].transAxes)
axs[2,1].text(.5, 1.05, 'Q interpolated unto BC grid', ha='center', transform=axs[2,1].transAxes)

plt.savefig('/home/bing/Downloads/BC_mod.png', bbox_inches='tight')
plt.close('all')
plt.ion()
'''