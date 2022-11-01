import slcp_modules.processing as processing
from matplotlib.patches import Polygon
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import PIL
import urllib
from typing import List


def plot_box(bb: dict):
    """
    Display a bounding box on a global map.

    :param bb: Dictionary that defines the bounding box to display
    """
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    poly = Polygon([(bb['min_lon'], bb['min_lat']), (bb['min_lon'], bb['max_lat']),
                    (bb['max_lon'], bb['max_lat']), (bb['max_lon'], bb['min_lat'])],
                   facecolor=(0, 0, 0, 0.0), edgecolor='black', linewidth=3, linestyle='--')
    ax.add_patch(poly)

    plt.show()


def plot_point(lat: float, lon: float):
    """
    Display a red dot at a given lat/lon on a global map

    :param lat: Latitude of point
    :param lon: Longitude of point
    """
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    ax.scatter([lon], [lat], s=100, color='red')
    plt.show()


def plot_map(da: xr.DataArray, title: str):
    """
    Plot color mesh data on a map

    :param da: Xarray dataarray object containing "lat", and "lon" dimensions
    """
    bounds = (da.lon.min() - 25,
              da.lat.min() - 25,
              da.lon.max() + 25,
              da.lat.max() + 25)

    fig = plt.figure(figsize=(15, 10))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    ax.set_extent(bounds, ccrs.PlateCarree())

    x, y = np.meshgrid(da.lon, da.lat)
    mesh = ax.pcolormesh(x, y, da.values, vmin=np.nanmin(da.values),
                         vmax=np.nanmax(da.values), cmap='jet')
    fig.colorbar(mesh, fraction=0.0284, pad=0.02)
    plt.title(title, fontsize=16)
    plt.show()


def comparison_plot(data: List[xr.DataArray], x_label: str, y_label: str, title: str = ''):
    """
    Plots multiple timeseries data on the same chart

    :param data: List of Xarray dataarrays that each contain 'time' dims
                    and a 'shortname' attr
    :param x_label: String for labelling the x-axis
    :param y_label: String for labelling the y-axis
    :param title: String for titelling the plot
    """
    plt.figure(figsize=(15, 6))

    for da in data:
        plt.plot(da.time, da.values, linewidth=2, label=da.attrs['shortname'])

    plt.grid(which='major', color='k', linestyle='-', zorder=0, alpha=0.25)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=45)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.show()


def ssh_temp_plot(ssh_da: xr.DataArray, temp_da: xr.DataArray):
    '''
    Vertically stacked plots comparing ECCO SSH time series with
    ECCO temperature time series at 15 depth levels
    '''
    from matplotlib import gridspec
    plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[.4, .6])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    p = ax2.pcolormesh(temp_da.time.values, temp_da.z.values, temp_da.data,
                       vmin=-1, vmax=1,
                       cmap='seismic', shading='gouraud')
    plt.colorbar(p, ax=ax2, orientation='horizontal',
                 shrink=.5, aspect=40, pad=0.15)
    ax2.set_ylabel('Depth (m)')
    ax2.set_xlabel('Time')
    ax2.grid(which='major', color='k', linestyle='-', alpha=0.25)
    ax2.set_title('ECCO Potential Temperature Anomaly (degrees C)')

    ax1 = plt.subplot(211, sharex=ax2)
    ax1.plot(ssh_da.time, ssh_da.values*0, 'k--', linewidth=3)
    ax1.plot(ssh_da.time, ssh_da.values, linewidth=2,
             label=ssh_da.attrs['shortname'])
    ax1.set_ylabel('SSH anomaly (m)')
    ax1.grid(which='major', color='k', linestyle='-', alpha=0.25)
    ax1.set_title('ECCO Sterodynamic SSH Anomaly (meters)')


def ipcc_plot(results: pd.DataFrame, subset: List[str] = [], name: str = ''):
    """
    Plot data from IPCC

    :param results: IPCC results in a Pandas DataFrame format
    :param subset: List of scenarios to plot
    :param name: Name of location
    """
    fig = plt.figure(figsize=(15, 5))

    cmap = plt.get_cmap("tab10")

    for i, (scenario, df) in enumerate(results.items()):
        color = cmap(i)

        if subset and scenario not in subset:
            continue

        plt.fill_between(df.year, df['height_17'], df['height_83'],
                         alpha=0.2, color=color)
        plt.plot(df.year, df['height_50'], label=scenario,
                 linewidth=3, color=color)

    plt.ylim(0.05, 2.5)
    plt.xlim(2020, 2150)
    plt.grid(which='major', color='k', linestyle='-', zorder=0, alpha=0.25)
    plt.legend(prop={'size': 12}, loc='upper left')
    plt.ylabel('Seal Level Change (m)', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    title = f'IPCC projection scenarios'
    title += f' at {name}' if name else ''
    title += '\nShaded ranges show 17th-83rd percentile ranges'
    plt.title(title, fontsize=16)
    plt.show()


def sea_plot(data: dict, rolling: bool = False):
    """
    Plot data from SEA

    :param data: SEA data formatted as a dictionary of Xarray DataArrays
    :param rolling: Boolean for applying rolling average to data
    """

    fig = plt.figure(figsize=(15, 5))

    cmap = plt.get_cmap("Set2")
    # Plot altimetry
    da = data['Altimetry']
    if rolling:
        da = processing.rolling_avg(da)
    plt.plot(da.time, da.values, color=cmap(0), linewidth=2, label='Altimetry')

    # Plot Tide Gauge
    da = data['Tide Gauge']
    if rolling:
        da = processing.rolling_avg(da)
    plt.plot(da.time, da.values, color=cmap(1), linewidth=2,
             label='Tide Gauge')
    da = data['Tide Gauge Trend']
    plt.plot(da.time, da.values, color=cmap(2), linewidth=2,
             label='Tide Gauge Trend')

    plt.xlim(da.time.values[0], da.time.values[-1])
    plt.grid(which='major', color='k', linestyle='-', zorder=0, alpha=0.25)
    plt.xlabel(da.attrs['x-label'], fontsize=12)
    plt.ylabel(da.attrs['y-label'], fontsize=12)
    plt.xticks(rotation=45)
    plt.title(
        f'Altimetry and Tide Gauge Comparison\n{da.attrs["name"]}', fontsize=16)
    plt.legend(prop={'size': 12})
    plt.show()


def taskforce_plot(data, subset=[], rolling=False):
    fig = plt.figure(figsize=(15, 6))

    cmap = plt.get_cmap("Set2")

    obs_da = data['Observations']

    keys = [k for k in data.keys() if k in subset]

    for i, (k) in enumerate(keys):
        if k not in subset:
            continue
        color = cmap(i)

        if type(data[k]) == xr.DataArray:
            da = data[k]
            if rolling:
                da = processing.rolling_avg(da)
            plt.plot(da.time, da.values / 1000,
                     linewidth=2, label=k, color=color)
        elif "vlm" not in k.lower():
            ds = data[k]
            plt.fill_between(ds.time, ds[f'{k}_17'].values / 1000,
                             ds[f'{k}_83'].values / 1000, alpha=0.2, color=color)
            plt.plot(ds.time, ds[f'{k}_50'].values /
                     1000, linewidth=2, label=k, color=color)

    plt.hlines([0], xmin=[obs_da.time.values[0]], xmax=[2150],
               zorder=0, alpha=0.5, color='black', linestyles='dashed')

    plt.xlim(obs_da.time.values[0], 2150)
    plt.grid(which='major', color='k', linestyle='-', zorder=0, alpha=0.25)
    plt.xlabel(obs_da.attrs['x-label'], fontsize=12)
    plt.ylabel(obs_da.attrs['y-label'], fontsize=12)
    plt.xticks(rotation=45)
    plt.title(f'Sea Level Scenarios w/ Observations', fontsize=16)
    plt.legend(prop={'size': 12})
    plt.show()


def get_ecco_tiles(date):
    base_url = f'https://sealevel-dataservices.jpl.nasa.gov/onearth/wmts/epsg4326/all//SSH_ECCO_version4_release4/default/{date}/4km/2/'
    tiles = {}
    print(f'Getting tile imagery from {base_url}')
    for i in range(3):
        for j in range(5):
            if j not in tiles.keys():
                tiles[j] = list()

            url = f'{base_url}{i}/{j}.png'
            try:
                r = urllib.request.urlopen(url)
                data = np.array(PIL.Image.open(r))
                tiles[j].append(data)
            except Exception as e:
                print(e)
    cols = [np.concatenate(tiles[k]) for k in tiles.keys()]
    all_data = np.concatenate(cols, axis=1)
    return all_data


def plot_tiles(tiles, title=''):
    plt.imshow(tiles)
    plt.ylim(1200, 100)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
