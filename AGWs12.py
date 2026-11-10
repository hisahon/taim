'''
Version 5:
    - I updated the method for referencing the source and implemented a 
      Gaussian approach to model it. The source is now stationary in space, with 
      temporal variations accounted for by considering the lightning flash rate.

Version 6: 
    -I will change the plotting mechanism from imshow to pcolormesh.
    Also i move some plot routines to another file. Criado um arquivo chamado 
    plotfigs.py 
    - Here i changed the source from a gaussian distribuction to a cilindrical
    and now i am using an normalized area of the piston. 
    -Add some pave parameters text to better control the ending
    -Add a new figure that shows the integration on the boundaries thats
    located on the plotfigs.py
    
Version 7:
    - I implement the source grid directly on this code, 
    - It was implemented also a new variable to move some coeficientes into the 
    SOR function, the variable name is 'coefs'.
    - Here i will use the agw_sor2, thats use a different aproach to borders 
    with parameter edge_order =2
    
Version 8:
    - Trying to obtain the real TEC by ions concentration np_o
    - Deleting some lines long time ago commented
    - Changed the indexex search to searchsorted
    
Version 9:
    - Updated the way that enter the lightning data, using a dictionary
    - Created a new function AGW2 just to include gamma, mask_b variables into
      coefs list, the old function AGW is still there for older versions of AGWs
    - Insert a vertical profile of neutral atmosphere and ionosphere

Version 10:
    - Declaring variables explicity from sub-routines
    - Declaring new variable to report where the source was declared: sour_in
    - Add a new figure report with Neutral/Ionosphere Behavior
    - New figure to show behavior of vertical profile
    - Separate the AGWs and Density and Temperature reports
    - Final Neutra_Ionosphere figures was transfered to plotfigs support file
Version 11:
    - Transfer the final figures to plotfigs support file
    - Deleted old export report formated_text, keeping only the new
    - Change the report to include the height difference between the base/top of consideration
Versão 12:
    - Updating the form of the glow is visualised changing the glow=(np_0-np_amb)
    to d\dt glows, where glow_h is the time/horizontal and glow_v is time/vertical 
    
'''

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import glob, os, fnmatch, gc, pickle
# from math import degrees as deg, radians as rad 
# from scipy.ndimage import zoom, gaussian_filter

from SIM_VOL import np, np_i, nf, nq, lon, lat, alt, alt_i, mask_b, dr, dt
from SIM_VOL import dp_m, dq_m, df_m, lon_min, lon_max, lat_min, lat_max
from SIM_VOL import np_amb, rho_o, tn_o, r_g, nu_nn

import IONO as aiw
import agw_sor2 as agw
import plotfigs


# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from datetime import datetime, timedelta
from numpy import linspace, zeros, array, save, savez, load, empty, flipud, isin, fliplr, searchsorted
from numpy import log10, gradient, abs, isnan, sqrt, exp, rad2deg, radians

from comv import folder_codes, generate_time_vector, save_param, next_letter, trim_time_series
from comv import reg, gaussian_2d, trim_series2, load_data, time_estimation, hanning_2d, trim_series
from comv import start_time, end_time, ltsb, lnsb, gau2d_v2



#%% Initialization
plt.rcParams.update({'font.size': 8})
plt.rc('font', family='serif')
plt.ioff()
# plt.ion()
    
# global gamma,n_sor

 
# p is height
# f is lon
# q is lat

alt_t = 0
itp = abs(alt - alt_t).argmin()

wp_m, wf_m, wq_m, wp_o, wf_o, wq_o, dTin = [zeros((np, nf, nq)) for _ in range(7)]
pot = empty((np_i, nf, nq))



# Total Electron Content (TEC) list
tecs, neuts = [], []

glows = []
glow_h, glow_v = [], []
np_h, np_v = [], []
# Atmospheric variables
pn = r_g * rho_o * tn_o
rho_amb = rho_o
tn_amb = tn_o
pn_amb = pn
np_o = np_amb
np_t = np_o

# Kinematic variables (locations and coordinates)
keop, keof, keoq = [], [], []  # height, lon, lat
keopi, keofi, keoqi = [], [], []  # height, lon, lat

neutral_h = []
neutral_v = []

# # Set specific lat/lon coordinates
# kep = 350
# # kef, keq = -58., -25. #a
# kef, keq = -56., -27. #iii
# # kef, keq = -54., -25.2 #ii
# # kef, keq = -55.5., -26.9. #i
# # kef, keq = -53.94, -23.7 #b
# # kef, keq = -54.35, -24.64 #d
# # kef, keq = -55, -26
rgk = reg['iii']
kep, kef, keq = 350., rgk['lonp'], rgk['latp'] #iii
# Find the closest points for altitude, longitude, and latitude
ikp = abs(alt - kep).argmin()
ikf = abs(rad2deg(lon) - kef).argmin()
ikq = abs(rad2deg(lat) - keq).argmin()


ips = [ikp, ikf, ikq]

# Initialize other variables
i_t = 0
stop_now = True

rn=1
#%% Variables Folders
fkeo = f'{folder_codes}/banco_dados/simulation/keograms'
frep = f'{folder_codes}/banco_dados/simulation/reports'

pname = next_letter(frep)
horario_inicial = datetime.now()
time_run = horario_inicial.strftime('%Y%j%H%M')
time_process = start_time.strftime('%Y%j')

horario_formatado = f'{time_process}_{time_run}{pname}'

save_folder = f'{folder_codes}/banco_dados/simulation/outputs/{horario_formatado}'
fwp = f'{save_folder}/wp'
fgrad_np = f'{save_folder}/glow'
fnp = f'{save_folder}/np_o'

fvprof_neutral = f'{save_folder}/nvp'
fvprof_iono = f'{save_folder}/ivp'
fvertical = f'{save_folder}/updraft'

fbd = f'{save_folder}/{pname}_{horario_formatado}_bd'


# fgw = f'{save_folder}/fgw'
# fxp = f'{save_folder}/fxp'

# Source folder for loading data
source_folder = f'{folder_codes}/banco_dado5s/bolt/matrix_simulation'

data_saveDir = f'/media/hisashi/D/Banco_Dados/simulation/{horario_formatado}'
# os.makedirs(data_saveDir, exist_ok=True)





#%% Options
gamma=1.4; n_sor=4

# amp = 5.e-6 #best at 1e-5
amp = 2.e-5 #best at 3.e-6
# mask_b = mask_b*0 + 1
mbf = 3
# mask_b = (mask_b + mbf)/(mbf+1)
mask_b[:] = 1
mask_v = 1

n0, n1 = 200, 400
i0, i1 = 200, 250

in0, in1 = searchsorted(alt, [n0, n1])
ii0, ii1 = searchsorted(alt_i, [i0, i1])

tvec = generate_time_vector(start_time, end_time, step_seconds=10)

# vcoef = 5e-3
# hcoef = 1e-1

vcoef = 1.e+0
hcoef = 1.e+0

# vcoef = hcoef
resetSource = True
saveMatrices = True
generateFigures = True
calDenTemp = True

formatted_text = f'''Simulation Parameters:
-----------------------
- Name               : {pname}
- Output Name        : {horario_formatado}
- Script File        : {os.path.basename(__file__)}

Simulation Time:
----------------
- Initial           : {start_time}
- Final             : {end_time}
Spatial Domain:
---------------
- Altitude Range     : [{int(alt.min())}, {int(alt.max())}] km
- Longitude Range    : [{int(rad2deg(lon.min()))}, {int(rad2deg(lon.max()))}] deg
- Latitude Range     : [{int(rad2deg(lat.min()))}, {int(rad2deg(lat.max()))}] deg

Resolution and Dimensions:
--------------------------
- Radial Step (dr)   : {dr} km
- Longitude Step (dln): {round((rad2deg(lon.max()) - rad2deg(lon.min())) / wp_o.shape[1], 2)} deg
- Latitude Step (dlt): {round((rad2deg(lat.max()) - rad2deg(lat.min())) / wp_o.shape[2], 2)} deg
- Time Step (dt)     : {dt} s
- Grid hg,ln,lt,tm   : {np}, {nf}, {nq}, {len(tvec)}

Keogram Coordinates:
----------------------
- Height (hg)        : {kep} km
- Longitude (ln)     : {kef} deg
- Latitude (lt)      : {keq} deg
- Amp Multiplier     : {amp:.1e}
- Source Altitude    : {alt_t} km
- Int. Neutral       : {n0}, {n1}
- Int. Ionosphere    : {i0}, {i1}

Mask Parameters:
----------------
- Mask_B (Max/Min)   : {round(mask_b.max(), 2)}, {round(mask_b.min(), 2)}
- Mask_V             : {mask_v}
- vcoef, hcoef       : {vcoef:.1e}, {hcoef:.1e}
- Iterations (n-SOR) : {n_sor}

Simulation Options:
-------------------
- Reset Source       : {resetSource}
- Save Matrices      : {saveMatrices}
- Generate Figures   : {generateFigures}
- Density/Temperature Variations: {calDenTemp}
'''

coefs = [vcoef, hcoef, n_sor, gamma, mask_b]
print(formatted_text)
save_param(formatted_text, filename=f'{frep}/{horario_formatado}.txt')

sour_in = zeros((nf, nq))

#%% Loading Data 

regions = {
    'a': {'use': False, 'site': 'a', 'folder': 'a_2500_5800', 'grad_on': True},
    'b': {'use': False, 'site': 'b', 'folder': 'b_2370_5394', 'grad_on': True},
    'c': {'use': False, 'site': 'c', 'folder': 'c_2419_5407', 'grad_on': True},
    'd': {'use': False, 'site': 'd', 'folder': 'd_2464_5435', 'grad_on': True},
    'i': {'use': False, 'site': 'i', 'folder': 'folder_i', 'grad_on': True},
    'ii': {'use': False, 'site': 'ii', 'folder': 'folder_ii', 'grad_on': False},
    'iii': {'use': True, 'site': 'iii', 'folder': 'folder_iii', 'grad_on': False},
}

# Initialize dictionaries to store results for each region
pdfs, tmss, itts = {}, {}, {}

# Iterate over all regions and process them

for region, params in regions.items():
    if params['use']:
        pdfs[region], _, _ = gau2d_v2(rad2deg(lon), rad2deg(lat), site=params['site'])
        # pdfs[region], _, _ = hanning_2d(rad2deg(lon), rad2deg(lat), site=params['site'])
        tmss[region], itts[region] = load_data(folder_codes, params['folder'], params['grad_on'])
        tmss[region], itts[region] = trim_series(tvec, tmss[region], itts[region], params['grad_on'])

# Find indices for latitude and longitude ranges
ilnn, ilnx = abs(lnsb - lon_min).argmin(), abs(lnsb - lon_max).argmin()
iltn, iltx = abs(ltsb - lat_min).argmin(), abs(ltsb - lat_max).argmin()



# # Set the extent (spatial boundaries)
# extd = [lon_min, lon_max, lat_min, lat_max]

sourcetime =[]

#%% Running
for it, timep in enumerate(tvec):
    
    if resetSource:
        sour_in[:] = 0
        
    ptime = datetime(2023, 11, 23) + timedelta(seconds = float(timep))
    
    #inserting the data inside the wp on the ground
    for region, params in regions.items():
        
        if params['use']:
            # print(region, params['use'], isin(timep, tmss[region]))
            if isin(timep, tmss[region]):
                t_idx = searchsorted(tmss[region], timep)
                sour_in += pdfs[region] * itts[region][t_idx] * amp
                # print(region, itts[region][t_idx] * amp)
        # print(sour_in.max())
            # else:
            #     sour_in += pdfs[region] * 0.
                
            # print(abs(sour_in).max())
            
    if it > 150:
        sour_in[:] = 0
            
    sourcetime.append(sour_in*1e3)
    wp_o[itp,:,:] = sour_in
    
    print ('======================================================')

    w_o = [wp_o, wq_o, wf_o]
    w_m = [wp_m, wq_m, wf_m]
    dts = [dp_m, mask_v*dq_m, mask_v*df_m, dt, dTin]
    amb_v = [pn_amb, rho_amb, rho_o, r_g, tn_o, nu_nn]
    
    wp, wf, wq = agw.AGW_AMP3(w_o, w_m, dts, amb_v, coefs)
    rho, tn = agw.AGW_DENSITY_TEMPERATURE(w_o, [wp, wf, wq], dts, amb_v, coefs, calDenTemp)
    
    # f=agw.AGW_AMP(gamma, mask_b, w_o, w_m, dts, amb_v, coefs)
    # wp, wf, wq, rho, tn, temp_osci= f[0], f[1], f[2], f[3], f[4], f[5]
    
    # wo2 = wp_o + wf_o + wq_o
    # w2 = wp + wf + wq
    
    wp_m, wf_m, wq_m = wp_o, wf_o, wq_o
    wp_o, wf_o, wq_o = wp, wf, wq
    
    #%%% Keograms
    # neuts.append((wo2 - w2).sum(0))
    # neuts.append(w2.sum(0))
    # neuts.append(wp[in0, :, :])
    neutral_h.append(wp[in0, :, :])
    
    # keop.append(w2[:, ikf, ikq])
    # keop.append(w2[:, :, :].sum(2).sum(1))
    # keop.append(wp[:, ikf, ikq])
    neutral_v.append(wp[:, ikf, ikq])
    # keop.append(wp_o[:, ikf, ikq])
    # neuts.append(wp[in0:in1,:,:].sum(0))
    # neuts.append(wp.sum(0))
       
    dtn = tn - tn_o
    rho_o, tn_o = rho, tn

    dtemp = tn - tn_amb
    # dTin  = dTin*0

    wp_i=wp[np-np_i: ,:, :]
    wf_i=wf[np-np_i:, :, :]
    wq_i=wq[np-np_i:, :, :]
    
    
    np_t = np_o
    np_o = aiw.den_evolve2(np_o, wp_i, wf_i, wq_i) # salvar np_o, gradient np_o = delta n
                                                   # integrar dn (altitude)  =  dTEC
    np_h.append(np_o[ii0:ii1,:,:].sum(0))
    np_v.append(np_o[:,ikf,ikq])
    
    glow=(np_o - np_amb)
    dnp_o = gradient(glow, axis=0, edge_order=2)
    
    # keopi.append(glow[:, ikf, ikq])
    
    # npkeoi = glow[ii0:ii1,:,:].sum(0)
    # npkeoi = glow[ii0:ii1,:,:].sum(0)
    # npkeoi = glow[ii1,:,:]
    # tecs.append(npkeoi)
    
#%%% Epoch Figures 
    if (it % 6) == 0 and generateFigures:
        ext = [lon_min, lon_max, lat_min, lat_max]
        plotfigs.plot_vertical_velocity_neutral(wp_o, ptime, i_t, ext, alt, fwp, mask_b)
        # plotfigs.plot_vertical_velocity_neutral(wp_o, processing_time, i_t, ext, alt, fwp, mask_v)
        # plotfigs.plot_special_frames(wp, sour_in, wp_o, ikp200, ikp300, ikp350, processing_time, i_t, ext, fxp)
        # plotfigs.plot_vertical_velocity_ionosphere(dnp_o, processing_time, i_t, ext, alt_i, fnp)
        plotfigs.plot_vertical_velocity_ionosphere(dnp_o, ptime, i_t, ext, alt_i, fgrad_np)
        # plotfigs.plot_vertical_info(wp, sour_in, mask_b, lat, lon, alt, processing_time, i_t, fhg)
        # plotfigs.plot_boundaries(wp, lat, lon, processing_time, i_t, fbd)
        plotfigs.plot_vertical_profile(rad2deg(lat), rad2deg(lon), alt, alt_i, wp, dnp_o, ips, ptime, fvertical)
                              
        plt.close('all')
    
    if saveMatrices:
        os.makedirs(f'{data_saveDir}', exist_ok=True)
        savez(f'{data_saveDir}/{timep}.npz',
              lts = rad2deg(lat),
              lns= rad2deg(lon),
              alt = alt,
              alt_i = alt_i,
              wp = array(wp),
              np_o = array(np_o)
              # glow = array(glow)
              )
        
    print (f"{pname} TIME = {it+1} [{int(i_t)}] iterations, dt = {dt}, {ptime}")
    print (f'Source min/max: {sour_in.min():.1e}, {sour_in.max():.1e}' )
    print (f'AGWs AMPLITUDES (wp, wf, wq) = ({wp.max():.1e}, {wf.max():.1e}, {wq.max():.1e})') 
    # print (f'GLOW LIM: (min, max, mean) = ({glow.min():.1e}, {glow.max():.1e}, {abs(glow).mean():.1e})')
    print (f'Progress: {it+1}/{len(tvec)}, {100*(it+1)/len(tvec):.2f} %, Est. Time: {time_estimation(horario_inicial, it, tvec)}')

    i_t += dt*rn
    # gc.collect()
           
# keop = array(keop)
# keopi = array(keopi)
# neuts = array(neuts)
# tecs = array(tecs)

neutral_h = array(neutral_h)
neutral_v = array(neutral_v)

glow_h = gradient(array(np_h), axis=0, edge_order=2)
glow_v = gradient(array(np_v), axis=0, edge_order=2)

sourcetime = array(sourcetime)

if saveMatrices:
    os.makedirs(f'{save_folder}', exist_ok=True)
    savez(f'{save_folder}/{horario_formatado}.npz',
         lts = rad2deg(lat),
         lns = rad2deg(lon),
         tvec = tvec,
         alt = alt,
         alt_i = alt_i,
         neutral_h = neutral_h,
         neutral_v = neutral_v,
         glow_h = glow_h,
         glow_v = glow_v
         )
    print(f'Saved DATAPLOT {horario_formatado}.npy')


#%% Comparison Figures
# plotfigs.plot_neutral_ionosphere_contourf(keop, keopi, neuts, tecs, tvec, alt, lat, lon, alt_i, regions, tmss, itts, amp, ikf, ikq, formatted_text)
plotfigs.plot_neutral_ionosphere_imshow(neutral_v, glow_v, neutral_h, glow_h, tvec, alt, alt_i, tmss, itts, lat, lon, regions, amp, ikf, ikq, formatted_text, fkeo, horario_formatado)

# plotfigs.plot_neutral_ionosphere_imshow(keop, keopi, sourcetime, tecs, tvec, alt, alt_i, tmss, itts, lat, lon, regions, amp, ikf, ikq, formatted_text, fkeo, horario_formatado)
#%% Source keogram

# # Create figure and axes
# fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
# tplt = tvec/3600

# axs = axs.flatten()

# # Neutral Atmosphere - Left Column
# # vm_neut_vert = abs(keop).max() * 1.0
# vmt = abs(sourcetime.sum(1)).max()
# vmb = abs(sourcetime.sum(2)).max()
# # vm_tec_vert = abs(keopi).max() * 1.0
# # vm_tec = (abs(tecs[:, ikf, :]).max() + abs(tecs[:, :, ikq]).max()) / 2 * 1.0

# # Row 1: Height vs Time (Neutral Atmosphere)
# ax = axs[0]
# ax.set_ylabel('lat, deg')
# ext = [tplt.min(), tplt.max(), rad2deg(lat).min(), rad2deg(lat).max()]
# ax.imshow(sourcetime.sum(1).T, origin='lower', extent=ext, interpolation='bilinear', aspect='auto', cmap='seismic_r', vmin = -vmt, vmax = vmt)
# ax.grid(True)
# # ax.set_ylim(0, 350)
# ax2 = ax.twinx()
# for region, params in regions.items():
#     if params['use']:
#         print(region, tmss[region]/3600, itts[region])
#         ax2.plot(tmss[region] / 3600, itts[region] * amp, lw=1, label=region)

# # ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='y = 150 km')
# # ax2_ylim = ax2.get_ylim()  # Get the y-axis limits of the main axis
# # ax2.set_ylim(ax2_ylim[0], 10*ax2_ylim[1])  # Set ax2 to occupy only the bottom half
# ax2.legend()

# # Row 1: Height vs Time (Ionosphere)
# ax = axs[1]
# ax.set_ylabel('lon, deg', color='red')
# ext = [tplt.min(), tplt.max(), rad2deg(lon).min(), rad2deg(lon).max()]
# ax.imshow(sourcetime.sum(2).T, origin='lower', extent=ext, interpolation='bilinear', aspect='auto', cmap='seismic_r', vmin = -vmb, vmax = vmb)
# # ax.text(1.08, 1., formatted_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
# #         family='monospace', bbox=dict(facecolor='none', alpha=0.2, edgecolor='none', boxstyle='round,pad=1'))
# ax.grid(True)
# # ax.set_ylim(0, 400)

# ax2 = ax.twinx()
# for region, params in regions.items():
#     if params['use']:
#         ax2.plot(tmss[region] / 3600, itts[region] * amp, lw=1, label=region)
        
# # ax2_ylim = ax2.get_ylim()  # Get the y-axis limits of the main axis
# # ax2.set_ylim(ax2_ylim[0], 10*ax2_ylim[1])  # Set ax2 to occupy only the bottom half
# # ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
# ax2.legend()


# # Save and show the figure
# # plt.savefig(f"{fkeo}/{horario_formatado}.png", bbox_inches='tight', dpi=100)
# plt.show()
# print('FIGURE SAVED')


























