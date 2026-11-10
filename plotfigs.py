# -*- coding: utf-8 -*-
#==============================================================================
#--- @author: Honda, R. H., on: Fri Oct  4 16:44:46 2024
#==============================================================================

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



def setup_ax(ax, extd):
    ax.coastlines(resolution='110m', color='k')
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent(extd, crs=ccrs.PlateCarree())
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle=':')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # gl.xlocator = MaxNLocator(integer=True)
    # gl.ylocator = MaxNLocator(integer=True)


# def plot_vertical_velocity_neutral(wp_o, processing_time, i_t, ext, alt, fwp):
#     """Plot vertical velocity of neutral atmosphere."""
#     fig, axs = plt.subplots(3, 3, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
#     axs = axs.flatten()
#     fig.suptitle(f'Vertical Velocity of Neutral Atmosphere [{processing_time}]  [{i_t}]')

#     cmt = int((wp_o.shape[0] - 1) / 8)
#     for i_plot in range(9):
#         ax = axs[i_plot]
#         data = wp_o[cmt * i_plot, :, :]
#         vm = abs(data).max() * 0.9

        
#         setup_ax(ax, ext)

#         ax.imshow(data.T, origin='lower', extent=ext, cmap='seismic', vmin=-vm, vmax=vm)
#         ax.grid(True)
#         ax.set_title(f'h={round(alt[cmt * i_plot])}, max*0.9 = {vm:.2e}')




#     plt.savefig(f"{fwp}/wp{processing_time.strftime('%H%M%S')}.png", bbox_inches='tight', dpi=150)
#     plt.close(fig)
    
def plot_vertical_velocity_neutral(wp_o, processing_time, i_t, ext, alt, fwp, mask_b):
    
    plt.ioff()
    os.makedirs(fwp, exist_ok=True)
    """Plot vertical velocity of neutral atmosphere."""
    
    aidxs = np.searchsorted(alt, [10, 100, 200, 300])
    
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()
    # fig.suptitle(f'Vertical Velocity of Neutral Atmosphere [{processing_time}]  [{i_t}]')
    
    

    cmt = int((wp_o.shape[0] - 1) / 4)
    vels = [0.1, 1, 25, 2]
    # vels = [0.1, 2, 30, 5] #2 sources
    for i_plot in range(4):
        ax = axs[i_plot]
        data = wp_o[aidxs[i_plot], :, :]
        # datap = data
        # array_log = np.log10(datap + 1e-5)
        vm = abs(data).max() * 1.001

        
        setup_ax(ax, ext)

        # ax.imshow(data.T, origin='lower', extent=ext, cmap='seismic_r', interpolation='bicubic', vmin=-vm, vmax=vm)
        ax.imshow(data.T, origin='lower', extent=ext, cmap='PiYG_r', interpolation='bicubic', vmin=-vm, vmax=vm)
        
        # ax.contourf(tvec/3600, alt_i, keopi.T, cmap='seismic_r', vmin = -vm, vmax = vm, levels = 100)
        # ax.imshow(mask_b[0].T, origin='lower', extent=ext, cmap='Greens_r', alpha = 0.3)
        ax.grid(True)
        ax.set_title(f'{chr(97+i_plot)}) Height={round(alt[aidxs[i_plot]])} km, {processing_time.strftime("%H:%M")} UT, {vels[i_plot]:.1e} m/s')
        
        # ax.set_title(f'{chr(97 + i_plot)}) height={round(alt_i[aidx])} km, [{processing_time.strftime("%H:%M")} UT]', fontsize=10)
        
        # Ajustando o tamanho da fonte dos rótulos dos eixos
        ax.set_xlabel('Longitude', fontsize=10)  # Rótulo do eixo x
        ax.set_ylabel('Latitude', fontsize=10)  # Rótulo do eixo y



    plt.savefig(f"{fwp}/wp{processing_time.strftime('%H%M%S')}.png", bbox_inches='tight', dpi=150)
    
    plt.close(fig)
    
    del fig, axs
    gc.collect()
    
def plot_vertical_profile(xlat, xlon, alt, alt_i, wp, glow, ipositions, ptime, folder):
    
    plt.ioff()
    os.makedirs(folder, exist_ok=True)
    """Plot vertical velocity of neutral atmosphere."""
    # fig, axs = plt.subplots(1, 2, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))

    # fig.suptitle(f'Vertical Profile Neutral/Ionosphere [{ptime}]')
    axs = axs.flatten()
    # fig.suptitle(f'Vertical Velocity of Neutral Atmosphere [{processing_time}]  [{i_t}]')
    
    jn, jf, jq = ipositions
    
    
    wplim = (abs(wp[:, jf, :]).max() + abs(wp[:, :, jq]).max())/2
    glim = (abs(glow[:, jf, :]).max() +abs(glow[:, :, jq]).max())/2
    
    ax = axs[0]
    # vm = abs(array3d[:, jf, :]).max()
    ax.set_title(f'a) Atmosphere, {ptime.strftime("%H:%M")} UT, {wplim:.1e} m/s')
    ax.contourf(xlat, alt, wp[:, jf, :], cmap='PiYG_r', levels = 50, vmin = -wplim, vmax = wplim)
    ax.set_ylabel('Height, km', color='black', fontsize=10)
    ax.set_xlabel('Latitude, deg', color='black', fontsize=10)
    ax.grid(True)
    
    ax = axs[2]
    # vm = abs(array3d[:, :, jq]).max()
    ax.contourf(xlon, alt, wp[:, :, jq], cmap='PiYG_r', levels = 50, vmin = -wplim, vmax = wplim)
    ax.set_title(f'c) Atmosphere, {ptime.strftime("%H:%M")} UT, {wplim:.1e} m/s')
    # ax.set_xlabel('(Neutral) time, h [UT]', fontsize=8, color='black', fontweight='bold')
    ax.set_ylabel('Height, km', color='black', fontsize=10)
    ax.set_xlabel('Longitude, deg', color='black', fontsize=10)
    ax.grid(True)
    
    ax = axs[1]
    # vm = abs(array3d[:, jf, :]).max()
    ax.contourf(xlat, alt_i, glow[:, jf, :], cmap='seismic_r', levels = 50, vmin = -glim, vmax = glim)
    ax.set_title(f'b) Ionosphere, {ptime.strftime("%H:%M")} UT, 20 %')
    # ax.set_ylabel('Height, km', color='black', fontsize=10)
    ax.set_xlabel('Latitude, deg', color='black', fontsize=10)
    ax.set_ylim(0,400)
    ax.grid(True)
    
    ax = axs[3]
    # vm = abs(array3d[:, :, jq]).max()
    ax.contourf(xlon, alt_i, glow[:, :, jq], cmap='seismic_r', levels = 50, vmin = -glim, vmax = glim)
    # ax.set_ylabel('Longitude, deg', color='red', fontsize=10)
    ax.set_ylim(0,400)
    ax.set_title(f'd) Ionosphere, [{ptime.strftime("%H:%M")} UT], 20 %')
    ax.set_ylabel('Height, km', color='black', fontsize=10)
    ax.set_xlabel('Longituder, deg', color='black', fontsize=10)
    ax.grid(True)


    # ax.set_title(f'h=vertical profile')


    plt.savefig(f"{folder}/updraft_{ptime.strftime('%H%M%S')}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    del fig, axs
    gc.collect()
    

def plot_special_frames(wp, sour_in, wp_o, ikp200, ikp300, ikp350, processing_time, i_t, ext, fxp):
    """Plot special frames."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()
    # ext = [lon_min, lon_max, lat_min, lat_max]
    fig.suptitle(f'Special Frames [{processing_time}]  [{i_t}]')

    # Ground velocity
    ax = axs[0]
    setup_ax(ax, ext)
    data = wp[0, :, :]  # Velocity at ground
    vm = abs(data).max() * 0.8
    ax.imshow(data.T, origin='lower', extent=ext, cmap='seismic', vmin=-vm, vmax=vm)
    ax.set_title(f'wp in the ground, max = {vm:.2e}')
    ax.grid(True)

    # Source point temperature
    ax = axs[1]
    setup_ax(ax, ext)
    data = sour_in
    vm = abs(data).max() * 0.8
    ax.imshow(data.T, origin='lower', extent=ext, cmap='seismic', vmin=-vm, vmax=vm)
    ax.set_title(f'dT sourcepoint, max = {vm:.2e}')
    ax.grid(True)

    # Integrated from 200 to 300 km
    ax = axs[2]
    setup_ax(ax, ext)
    data = wp_o[ikp200:ikp300, :, :].sum(0)  # TEC from 200 to 300 km
    vm = abs(data).max() * 0.8
    ax.imshow(data.T, origin='lower', extent=ext, cmap='seismic', vmin=-vm, vmax=vm)
    ax.set_title(f'Integrated from 200 up to 300 km, max = {vm:.2e}')
    ax.grid(True)

    # Integrated from 200 to 350 km
    ax = axs[3]
    setup_ax(ax, ext)
    data = wp_o[ikp200:ikp350, :, :].sum(0)  # TEC from 200 to 350 km
    vm = abs(data).max() * 0.8
    ax.imshow(data.T, origin='lower', extent=ext, cmap='seismic', vmin=-vm, vmax=vm)
    ax.set_title(f'Integrated from 200 up to 350 km, max = {vm:.2e}')
    ax.grid(True)

    plt.savefig(f"{fxp}/xp{processing_time.strftime('%H%M%S')}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    gc.collect()


def plot_vertical_velocity_ionosphere(data_in, processing_time, i_t, ext, alt_i, folder):
    """Plot vertical velocity of ionosphere."""
    
    plt.ioff()
    os.makedirs(folder, exist_ok=True)
    aidxs = np.searchsorted(alt_i, [100, 150, 250, 300])
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()
    
    #fig.suptitle(f'Vertical Velocity of Ionosphere at time [{processing_time}] [{i_t}]')

    for i_plot in range(4):
        ax = axs[i_plot]
        aidx = aidxs[i_plot]
        data = data_in[aidx, :, :]
        vm = abs(data_in).max()
        
        # cmt = int((glow.shape[0] - 1) / 3)
        # data = glow[cmt * i_plot + 0, :, :]
        # vm = abs(data).max()
        # vm = abs(data).max()
        setup_ax(ax, ext)
       
        ax.imshow(data.T, origin='lower', extent=ext, cmap='seismic_r', interpolation='bicubic', vmin=-vm, vmax=vm)
        
        ax.grid(True)
        # Definindo o título com tamanho de fonte personalizado
        ax.set_title(f'{chr(97 + i_plot)}) height={round(alt_i[aidx])} km, {processing_time.strftime("%H:%M")} UT, 15 %')
        
        # Ajustando o tamanho da fonte dos rótulos dos eixos
        ax.set_xlabel('Longitude, deg', fontsize=10)  # Rótulo do eixo x
        ax.set_ylabel('Latitude, deg', fontsize=10)  # Rótulo do eixo y

    plt.tight_layout()
    plt.savefig(f"{folder}/glow{processing_time.strftime('%H%M%S')}.png", bbox_inches='tight', dpi = 150)
    plt.close(fig)
    
    del fig, axs
    gc.collect()
    
def plot_keo_data(tvec, alt, lon, lat, keop, keof, keoq, tms_a, tms_b, tms_c,\
                  tms_d, itt_a, itt_b, itt_c, itt_d, kep, kef, keq, dt, mov_avg, fkeo):
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    axs = axs.flatten()

    # Parâmetros para a colormap
    rng = 10
    vm_keop = keop.max()
    vm_keof = keof.max() * 0.9
    vm_keoq = keoq.max() * 0.9

    # Plot para keop
    ax = axs[0]
    ax.set_title(f'Selected position: height = {kep} km, lon = {kef} deg, lat = {keq} deg, dt = {dt}, ma = {mov_avg}')
    ax.set_ylabel('height, km')
    ax.pcolormesh(tvec / 3600, alt, keop.T, shading='auto', cmap='seismic', vmin=-vm_keop, vmax=vm_keop)
    ax.plot(tms_a / 3600, 30 * itt_a, label='i')
    ax.plot(tms_b / 3600, 30 * itt_b, label='ii')
    ax.plot(tms_c / 3600, 30 * itt_c, label='iii')
    ax.plot(tms_d / 3600, 30 * itt_d, label='iv')
    ax.legend()
    ax.grid(True)

    # Plot para keof
    ax = axs[1]
    ax.set_ylabel('lon, deg')
    ax.pcolormesh(tvec / 3600, np.rad2deg(lon), keof.T, shading='auto', cmap='seismic', vmin=-vm_keof, vmax=vm_keof)
    ax.grid(True)

    # Plot para keoq
    ax = axs[2]
    ax.set_ylabel('lat, deg')
    ax.pcolormesh(tvec / 3600, np.rad2deg(lat), keoq.T, shading='auto', cmap='seismic', vmin=-vm_keoq, vmax=vm_keoq)
    ax.grid(True)

    # Definir o rótulo do eixo x no último gráfico
    axs[2].set_xlabel('time, h [ut]')

    # Salvar a figura com a data e hora atuais
    horario_atual = datetime.now()
    horario_formatado = horario_atual.strftime('%Y%m%d%H%M%S')
    plt.tight_laout()
    plt.savefig(f"{fkeo}/wp_keo_{horario_formatado}.png", dpi=150)
    

    plt.show()
    gc.collect()
    
def plot_vertical_info(wp, sour_in, mask_b, lat, lon, alt, processing_time, i_t, fhg):
    """
    Plots the vertical velocity of a neutral atmosphere in a 2x2 grid of subplots.

    Parameters:
        wp (np.ndarray): Array containing vertical velocity data.
        sour_in (np.ndarray): Source input data for plotting.
        mask_b (np.ndarray): Masking data for plotting.
        lat (np.ndarray): Latitude values.
        lon (np.ndarray): Longitude values.
        alt (np.ndarray): Altitude values.
        processing_time (datetime): Time of processing for title.
        i_t (int): Index for time or iteration.
        fhg (str): File path for saving the figure.
    """
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()
    fig.suptitle(f'Vertical Velocity of Neutral Atmosphere [{processing_time}]  [{i_t}]')

    # First subplot
    ax = axs[0]
    data = wp.sum(axis=1)
    vm = abs(data).max() * 1.00001
    ax.pcolormesh(np.rad2deg(lat), alt, data, cmap='seismic', vmin=-vm, vmax=vm)
    ax.plot(np.rad2deg(lat), mask_b[0, 50, :] * 100)
    ax.grid(True)
    ax.set_title(f'max = {vm:.2e}')

    # Second subplot
    ax = axs[1]
    data = wp.sum(axis=2)
    vm = abs(data).max() * 1.000001
    ax.pcolormesh(np.rad2deg(lon), alt, data, cmap='seismic', vmin=-vm, vmax=vm)
    ax.plot(np.rad2deg(lon), mask_b[0, :, 50] * 100)
    ax.grid(True)
    ax.set_title(f'max = {vm:.2e}')

    # Third subplot
    ax = axs[2]
    data = sour_in.sum(axis=0)
    ax.plot(np.rad2deg(lat), data)
    # ax.set_ylim(0, 1e-3)
    ax.grid(True)

    # Fourth subplot
    ax = axs[3]
    data = sour_in.sum(axis=1)
    ax.plot(np.rad2deg(lon), data)
    ax.grid(True)
    
    plt.savefig(f"{fhg}/high{processing_time.strftime('%H%M%S')}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    gc.collect()
    
def plot_boundaries(wp, lat, lon, processing_time, i_t, save_path):

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs = axs.flatten()
    fig.suptitle(f'Integrated boundaries [{processing_time}]  [{i_t}]', fontsize=10)

    # First plot: wp in height
    ax = axs[0]
    data_h0 = wp[0, :, :].sum(0)
    data_hmax = wp[-1, :, :].sum(0)
    
    # Plot the first dataset with blue dashed line
    ax.plot(np.rad2deg(lat), data_h0, label='h=0', linestyle='--', color='blue', linewidth=2)
    
    # Create twin y-axis for second dataset
    ax_twin = ax.twinx()
    ax_twin.plot(np.rad2deg(lat), data_hmax, label='h=max', linestyle='--', color='red', linewidth=2)
    
    # Set title, grid, labels, and legend
    ax.set_title('wp in the height', fontsize=12)
    ax.grid(True)
    # ax.set_xlabel('Latitude (degrees)')
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')

    # Second plot: wp in longitude
    ax = axs[1]
    data_h0 = wp[:, 0, :].sum(0)
    data_hmax = wp[:, -1, :].sum(0)
    
    # Plot first dataset with blue dashed line
    ax.plot(np.rad2deg(lat), data_h0, label='lon=0', linestyle='--', color='blue', linewidth=2)
    
    # Create twin y-axis for second dataset
    ax_twin = ax.twinx()
    ax_twin.plot(np.rad2deg(lat), data_hmax, label='lon=max', linestyle='--', color='red', linewidth=2)
    
    # Set title, grid, labels, and legend
    ax.set_title('wp in the longitudes', fontsize=12)
    ax.grid(True)
    # ax.set_xlabel('Latitude (degrees)')
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')

    # Third plot: wp in latitude
    ax = axs[2]
    data_h0 = wp[:, :, 0].sum(0)
    data_hmax = wp[:, :, -1].sum(0)
    
    # Plot first dataset with blue dashed line
    ax.plot(np.rad2deg(lon), data_h0, label='lat=0', linestyle='--', color='blue', linewidth=2)
    
    # Create twin y-axis for second dataset
    ax_twin = ax.twinx()
    ax_twin.plot(np.rad2deg(lon), data_hmax, label='lat=max', linestyle='--', color='red', linewidth=2)
    
    # Set title, grid, labels, and legend
    ax.set_title('wp in the latitudes', fontsize=12)
    ax.grid(True)
    # ax.set_xlabel('Longitude (degrees)')
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')

    # Save the figure
    fig.savefig(f"{save_path}/bd{processing_time.strftime('%H%M%S')}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    gc.collect()


# import numpy as np
# import matplotlib.pyplot as plt
# from math import radians, degrees

def plot_neutral_ionosphere_imshow(keop, tec_v, neuts, tec_h, tvec, alt, alt_i, tmss, itts, lat, lon, cases, amp, ikf, ikq, formatted_text, fkeo, horario_formatado):
    """
    Generates a 3x2 grid plot comparing neutral atmosphere and ionosphere data.
    
    Parameters:
        keop: 2D array, neutral atmosphere height vs time data.
        neuts: 3D array, neutral atmosphere latitude and longitude data.
        tecs: 3D array, ionosphere latitude and longitude data.
        tvec: 1D array, time vector in seconds.
        alt: 1D array, altitude vector in km.
        lat: 1D array, latitude vector in radians.
        lon: 1D array, longitude vector in radians.
        cases: dict, dictionary containing case-specific parameters.
        amp: float, amplitude scaling factor for twin-axis plots.
        ikf: int, index for latitude slice in tecs.
        ikq: int, index for longitude slice in tecs.
        formatted_text: str, text to display on the plot.
        fkeo: str, folder path for saving the figure.
        horario_formatado: str, formatted timestamp for saving the figure.
    """
    # Create figure and axes
    fig, axs = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    axs = axs.flatten()

    # Neutral Atmosphere - Left Column
    vm_neut_vert = abs(keop).max() * 1.0
    vm_neut = (abs(neuts.sum(1)).max() + abs(neuts.sum(2)).max()) / 2 * 1.
    vm_tec_vert = abs(tec_v).max() * 1.0
    # vm_tec = (abs(tec_h[:, ikf, :]).max() + abs(tec_h[:, :, ikq]).max()) / 2 * 1.0
    
    vm_tec = (abs(tec_h.sum(1)).max() + abs(tec_h.sum(2)).max()) / 5 * 1.

    # Row 1: Height vs Time (Neutral Atmosphere)
    ax = axs[0]
    ax.set_ylabel('Height, km')
    ext = [tvec.min() / 3600, tvec.max() / 3600, alt.min(), alt.max()]
    ax.imshow(keop.T, origin='lower', extent=ext, interpolation='bilinear', aspect='auto', cmap='seismic_r',
              vmin=-vm_neut_vert, vmax=vm_neut_vert)
    ax.grid(True)
    ax.set_ylim(0, 400)
    ax2 = ax.twinx()
    for case, params in cases.items():
        if params['use']:
            ax2.plot(tmss[case] / 3600, itts[case] * amp, lw=1, label=f"site: {params['site']}")
    
    
    ax2_ylim = ax2.get_ylim()  # Get the y-axis limits of the main axis
    ax2.set_ylim(ax2_ylim[0], 10*ax2_ylim[1])  # Set ax2 to occupy only the bottom half
    ax2.legend()

    # Row 2: Latitude vs Time (Neutral Atmosphere)
    ax = axs[2]
    ax.set_ylabel('Latitude, deg')
    ext = [tvec.min() / 3600, tvec.max() / 3600, np.degrees(lat).min(), np.degrees(lat).max()]
    ax.imshow(neuts.sum(1).T, origin='lower', extent=ext, interpolation='bilinear', aspect='auto', cmap='seismic_r',
              vmin=-vm_neut, vmax=vm_neut)
    
    
    # ax.set_ylim()
    ax.grid(True)

    # Row 3: Longitude vs Time (Neutral Atmosphere)
    ax = axs[4]
    ax.set_ylabel('Longitude, deg')
    ext = [tvec.min() / 3600, tvec.max() / 3600, np.degrees(lon).min(), np.degrees(lon).max()]
    ax.imshow(neuts.sum(2).T, origin='lower', extent=ext, interpolation='bilinear', aspect='auto', cmap='seismic_r',
              vmin=-vm_neut, vmax=vm_neut)
    ax.grid(True)
    ax.set_xlabel('(NEUTRAL ATMOSPHERE) time, h [UT]', fontsize=8, color='black', fontweight='bold')

    # Row 1: Height vs Time (Ionosphere)
    ax = axs[1]
    ax.set_ylabel('Height, km', color='red')
    ext = [tvec.min() / 3600, tvec.max() / 3600, alt_i.min(), alt_i.max()]
    ax.imshow(tec_v.T, origin='lower', extent=ext, interpolation='bilinear', aspect='auto', cmap='seismic_r',
              vmin=-vm_tec_vert, vmax=vm_tec_vert)
    ax.text(1.08, 1., formatted_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            family='monospace', bbox=dict(facecolor='none', alpha=0.2, edgecolor='none', boxstyle='round,pad=1'))
    ax.grid(True)
    ax.set_ylim(0, 400)
    
    ax2 = ax.twinx()
    for case, params in cases.items():
        if params['use']:
            ax2.plot(tmss[case] / 3600, itts[case] * amp, lw=1, label=case)
            
    ax2_ylim = ax2.get_ylim()  # Get the y-axis limits of the main axis
    ax2.set_ylim(ax2_ylim[0], 10*ax2_ylim[1])  # Set ax2 to occupy only the bottom half
    ax2.legend()

    # Row 2: Latitude vs Time (Ionosphere)
    ax = axs[3]
    ax.set_ylabel('Latitude, deg', color='red')
    ext = [tvec.min() / 3600, tvec.max() / 3600, np.degrees(lat).min(), np.degrees(lat).max()]
    ax.imshow(tec_h.sum(1).T, origin='lower', extent=ext, interpolation='bilinear', aspect='auto', cmap='seismic_r',
              vmin=-vm_tec, vmax=vm_tec)
    ax.grid(True)

    # Row 3: Longitude vs Time (Ionosphere)
    ax = axs[5]
    ax.set_ylabel('Longitude, deg', color='red')
    ext = [tvec.min() / 3600, tvec.max() / 3600, np.degrees(lon).min(), np.degrees(lon).max()]
    ax.imshow(tec_h.sum(2).T, origin='lower', extent=ext, interpolation='bilinear', aspect='auto', cmap='seismic_r',
              vmin=-vm_tec, vmax=vm_tec)
    ax.set_xlabel('(IONOSPHERE) time, h [UT]', fontsize=8, color='red', fontweight='bold')
    ax.grid(True)

    # Save and show the figure
    plt.savefig(f"{fkeo}/{horario_formatado}.png", bbox_inches='tight', dpi=100)
    plt.show()
    print('FIGURE SAVED')

# import numpy as np
# import matplotlib.pyplot as plt
# from math import radians, degrees

def plot_neutral_ionosphere_contourf(
    keop, keopi, neuts, tecs, tvec, alt, lat, lon, alt_i, cases, tmss, itts, amp, ikf, ikq, formatted_text
):
    """
    Generates a 3x2 grid plot comparing neutral atmosphere and ionosphere data using contourf.
    
    Parameters:
        keop: 2D array, neutral atmosphere height vs time data.
        neuts: 3D array, neutral atmosphere latitude and longitude data.
        tecs: 3D array, ionosphere latitude and longitude data.
        tvec: 1D array, time vector in seconds.
        alt: 1D array, altitude vector in km.
        lat: 1D array, latitude vector in radians.
        lon: 1D array, longitude vector in radians.
        alt_i: 1D array, ionosphere altitude vector in km.
        cases: dict, dictionary containing case-specific parameters.
        amp: float, amplitude scaling factor for twin-axis plots.
        ikf: int, index for latitude slice in tecs.
        ikq: int, index for longitude slice in tecs.
        formatted_text: str, text to display on the plot.
    """
    # Create figure and axes
    fig, axs = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    axs = axs.flatten()

    # Neutral Atmosphere - Left Column
    vm_neut_vert = abs(keop).max() * 1.0
    vm_neut = (neuts.sum(1).max() + neuts.sum(2).max()) / 2 * 1.01
    vm_tec_vert = abs(keopi).max() * 1.0
    vm_tec = abs(tecs).max() * 1.0

    # Row 1: Height vs Time (Neutral Atmosphere)
    ax = axs[0]
    ax.set_ylabel('Height, km')
    ax.contourf(tvec / 3600, alt, keop.T, cmap='seismic_r', vmin=-vm_neut_vert, vmax=vm_neut_vert, levels=100)
    ax.grid(True)
    ax.set_ylim(0, 350)
    ax2 = ax.twinx()
    for case, params in cases.items():
        if params['use']:
            ax2.plot(tmss[case] / 3600, itts[case] * amp, lw=1, label=f"site: {params['site']}")
    
    ax2_ylim = ax2.get_ylim()  # Get the y-axis limits of the main axis
    ax2.set_ylim(ax2_ylim[0], 10*ax2_ylim[1])  # Set ax2 to occupy only the bottom half
    ax2.legend()

    # Row 2: Latitude vs Time (Neutral Atmosphere)
    ax = axs[2]
    ax.set_ylabel('Latitude, deg')
    ax.contourf(tvec / 3600, np.degrees(lat), (neuts.sum(1)).T, cmap='seismic_r', vmin=-vm_neut, vmax=vm_neut, levels=100)
    ax.grid(True)

    # Row 3: Longitude vs Time (Neutral Atmosphere)
    ax = axs[4]
    ax.set_ylabel('Longitude, deg')
    ax.contourf(tvec / 3600, np.degrees(lon), (neuts.sum(2)).T, cmap='seismic_r', vmin=-vm_neut, vmax=vm_neut, levels=100)
    ax.grid(True)
    ax.set_xlabel('(NEUTRAL ATMOSPHERE) time, h [UT]', fontsize=8, color='black', fontweight='bold')

    # Row 1: Height vs Time (Ionosphere)
    ax = axs[1]
    ax.set_ylabel('Height, km', color='red')
    ax.contourf(tvec / 3600, alt_i, keopi.T, cmap='seismic_r', levels=100)
    ax.text(1.08, 1., formatted_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            family='monospace', bbox=dict(facecolor='none', alpha=0.2, edgecolor='none', boxstyle='round,pad=1'))
    ax.grid(True)
    ax.set_ylim(20, 400)
    ax2 = ax.twinx()
    for case, params in cases.items():
        if params['use']:
            ax2.plot(tmss[case] / 3600, itts[case] * amp, lw=1, label=case)
    ax2.legend()

    # Row 2: Latitude vs Time (Ionosphere)
    ax = axs[3]
    ax.set_ylabel('Latitude, deg', color='red')
    ax.contourf(tvec / 3600, np.degrees(lat), (tecs[:, ikf, :]).T, cmap='seismic_r', levels=50)
    ax.grid(True)

    # Row 3: Longitude vs Time (Ionosphere)
    ax = axs[5]
    ax.set_ylabel('Longitude, deg', color='red')
    ax.contourf(tvec / 3600, np.degrees(lon), (tecs[:, :, ikq]).T, cmap='seismic_r', levels=50)
    ax.set_xlabel('(IONOSPHERE) time, h [UT]', fontsize=8, color='red', fontweight='bold')
    ax.grid(True)

    # Save and show the figure
    plt.show()