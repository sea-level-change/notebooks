from typing import Callable, List

import numpy as np
import pandas as pd
import xarray as xr


def search_inventory(df: pd.DataFrame, col: str, term: str) -> List[str]:
    '''
    Search the titles of the contents in the inventory results.

    Term can be a multiworded string. Search will look for words in any order.

    Return: list of results
    '''
    terms = term.split()
    pattern = ''
    for t in terms:
        pattern += f'(?=.*{t})'

    matches = df[df[col].str.contains(
        pattern, case=False, regex=True, na=False)].iloc
    results = [r[col] for r in matches if '_clim' not in r[col]]

    if results:
        return results
    else:
        return f'Search term "{term}" not found in inventory {col}s'


def prep_ts(ts_json: dict, procs: List[Callable] = []) -> xr.DataArray:
    """
    Generates Xarray DataArray object from timeSeriesSpark algorithm results

    :param ts_json: JSON formatted timeSeriesSpark algorithm results
    :param proc: list of processing functions to be applied to data

    :return da: DataArray with data, a time coordinate, and a shortname attribute
    """
    shortname = ts_json['meta'][0]['shortName']
    time = np.array([np.datetime64(ts[0]["iso_time"][:10])
                    for ts in ts_json["data"]])
    das = []
    for var in ['mean', 'meanSeasonal']:
        vals = np.array([ts[0][var] for ts in ts_json["data"]])
        da = xr.DataArray(vals, coords=[time], dims=['time'], name=var)
        for proc in procs:
            da = proc(da)
        da.attrs['shortname'] = shortname
        das.append(da)

    ds = xr.merge(das)
    return ds


def prep_avg_map(var_json: dict, units: str = '') -> xr.DataArray:
    """
    Generates Xarray DataArray object from timeAvgMapSpark algorithm results

    :param var_json: JSON formatted timeAvgMapSpark algorithm results

    :return da: DataArray with data, lat/lon coordinates, and shortname and units attributes
    """
    shortname = var_json['meta']['shortName']

    vals = np.array([v['mean'] for var in var_json['data'] for v in var])
    lats = np.array([var[0]['lat'] for var in var_json['data']])
    lons = np.array([v['lon'] for v in var_json['data'][0]])

    vals[vals == -9999] = np.nan

    vals_2d = np.reshape(vals,
                         (len(var_json['data']),
                          len(var_json['data'][0])))

    da = xr.DataArray(vals_2d,
                      coords={"lat": lats, "lon": lons},
                      dims=["lat", "lon"])
    da.attrs['shortname'] = shortname
    da.attrs['units'] = units
    return da


def prep_ipcc(r: dict) -> dict:
    """
    Generates dictionary of Pandas DataFrames

    :param r: JSON formatted response from IPCC

    :return results: Dictionary with IPCC scenarios as keys, and values consisting of Pandas DataFrame 
                        containing year and percentile data
    """
    results = {}
    for result in r:
        df = pd.DataFrame(
            {k: result[k] for k in result.keys() if 'height' in k or 'year' in k})
        df.attrs = {'scenario': result['scenario']}
        results[result['scenario']] = df
    return results


def prep_sea(r: dict) -> dict:
    """
    Generates dictionary of Xarray DataArrays

    :param r: JSON formatted response from SEA

    :return data: Dictionary with data vars as keys, and values consisting of Xarray DataArrays
    """
    data = {}

    properties = r['features'][0]['properties']

    global_attrs = {
        'coords': r['features'][0]['geometry']['coordinates'],
        'name': properties['name'].title()
    }

    alt = pd.DataFrame(properties['altimetry_time_series'])
    data['Altimetry'] = xr.DataArray(alt.y,
                                     coords=[alt.x.values],
                                     dims=['time'],
                                     attrs={'x-label': alt['x-label'][0],
                                            'y-label': alt['y-label'][0],
                                            'coords': global_attrs['coords'],
                                            'name': global_attrs['name']})

    tide = pd.DataFrame(properties['tide_gauge_time_series'])
    data['Tide Gauge'] = xr.DataArray(tide.y,
                                      coords=[tide.x.values],
                                      dims=['time'],
                                      attrs={'x-label': tide['x-label'][0],
                                             'y-label': tide['y-label'][0],
                                             'coords': global_attrs['coords'],
                                             'name': global_attrs['name']})
    data['Tide Gauge Trend'] = xr.DataArray(tide['y-trend'].values,
                                            coords=[tide.x.values],
                                            dims=['time'],
                                            attrs={'x-label': tide['x-label'][0],
                                                   'y-label': tide['y-label'][0],
                                                   'coords': global_attrs['coords'],
                                                   'name': global_attrs['name']})
    data['Tide Gauge (subsidence corrected)'] = xr.DataArray(tide['y-subsidence-corrected'],
                                                             coords=[
                                                                 tide.x.values],
                                                             dims=['time'],
                                                             attrs={'x-label': tide['x-label'][0],
                                                                    'y-label': tide['y-label'][0],
                                                                    'coords': global_attrs['coords'],
                                                                    'name': global_attrs['name']})
    data['Tide Gauge Trend (subsidence corrected)'] = xr.DataArray(tide['y-trend-subsidence-corrected'].values,
                                                                   coords=[
                                                                       tide.x.values],
                                                                   dims=[
                                                                       'time'],
                                                                   attrs={'x-label': tide['x-label'][0],
                                                                          'y-label': tide['y-label'][0],
                                                                          'coords': global_attrs['coords'],
                                                                          'name': global_attrs['name']})

    return data


def prep_gmsl(r: dict) -> xr.DataArray:
    """
    Generates Xarray DataArray from indicator backend response

    :param r: JSON formatted response from indicator backend

    :return da: Xarray DataArray
    """
    da = xr.DataArray(r['response']['docs'][0]['y'],
                      coords=[r['response']['docs'][0]['x']],
                      dims=['time'])
    return da


def prep_taskforce(r: dict) -> dict:
    """
    Generates dictionary of Xarray DataArrays

    :param r: JSON formatted response from SEA

    :return data: Dictionary with data vars as keys, and values consisting of Xarray DataArrays
    """
    data = {}

    projections = []
    for result in r:
        if result['type'] == 'tide_gauge':
            if result['scenario'] == 'Trajectory':
                traj = result
            elif result['process'] == 'total':
                projections.append(result)

    data['Observations'] = xr.DataArray(traj['height_50'],
                                        coords=[traj['year']],
                                        dims=['time'],
                                        attrs={'x-label': 'Year',
                                               'y-label': 'Sea Level Change (m)',
                                               'coords': 'time',
                                               'name': 'Observations'})

    for projection in projections:
        ds = xr.Dataset(
            data_vars={
                f'{projection["scenario"]}_17': projection['height_17'],
                f'{projection["scenario"]}_50': projection['height_50'],
                f'{projection["scenario"]}_83': projection['height_83'],
            },
            coords={'time': projection['year']},
            attrs={'x-label': 'Year',
                   'y-label': 'Sea Level Change (m)',
                   'coords': 'time'}
        )
        data[projection['scenario']] = ds
    return data


def merge_depth_results(results: List[xr.DataArray]) -> xr.DataArray:
    '''
    Merges ECCO 3d results (15 depth levels) into a single dataarray
    '''
    ds = xr.concat(results, 'z')
    depths = [-5, -15, -25, -35, -45,
              -55, -65, -75, -85, -95.1,
              -105.31, -115.87, -127.15, -139.74, -154.47]
    ds = ds.assign_coords({'z': depths})
    return ds


def grac_mascon(da: xr.DataArray) -> xr.DataArray:
    da.values = (da.values / 9.81) / 10
    da_a = da - da.mean('time')
    da_clim = da_a.groupby("time.month").mean("time")
    da_clim_anom = da_a.groupby("time.month") - da_clim
    da_clim_anom_ra = da_clim_anom.rolling(time=3, center=True).mean()

    return da_clim_anom_ra


def ecco_grace(da: xr.DataArray) -> xr.DataArray:
    da.values = (da.values)
    da_a = da - da.mean('time')
    da_clim = da_a.groupby("time.month").mean("time")
    da_clim_anom = da_a.groupby("time.month") - da_clim
    da_clim_anom_ra = da_clim_anom.rolling(time=3, center=True).mean()

    return da_clim_anom_ra


def ecco_theta(da):
    da.values = (da.values)
    da_a = da - da.mean('time')
    da_clim = da_a.groupby("time.month").mean("time")
    da_clim_anom = da_a.groupby("time.month") - da_clim
    da_clim_anom_ra = da_clim_anom.rolling(time=3, center=True).mean()
    return da_clim_anom_ra


def rolling_avg(da: xr.DataArray, time: int = 3) -> xr.DataArray:
    '''
    Perform a rolling average on the "time" dimension of a DataArray
    '''
    avg = da.rolling(time=time, center=True).mean()
    avg.attrs = da.attrs
    return avg
