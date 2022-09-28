import time
import requests

dt_format = "%Y-%m-%dT%H:%M:%SZ"


def spatial_mean(base_url, dataset, bb, start_time, end_time):
    """
    Submit a GET request to data analysis tool to perform a timeseries analysis on the given dataset,
    spatial, and temporal parameters.

    :return response.json(): JSON formatted response
    """
    params = {
        'ds': dataset,
        'minLon': bb['min_lon'],
        'minLat': bb['min_lat'],
        'maxLon': bb['max_lon'],
        'maxLat': bb['max_lat'],
        'startTime': start_time.strftime(dt_format),
        'endTime': end_time.strftime(dt_format)
    }
    url = f'{base_url}/timeSeriesSpark'

    print(f'{url}?{"&".join(f"{k}={v}" for k,v in params.items())}')
    print("Waiting for response from data analysis tool...")
    start = time.perf_counter()
    response = requests.get(url, params=params, verify=False)
    print(f"Time series took {(time.perf_counter() - start):.3f} seconds\n")
    return response.json()


def time_avg_map(base_url, dataset, bb, start_time, end_time):
    """
    Submit a GET request to data analysis tool to perform a time average analysis on the given dataset,
    spatial, and temporal parameters. Results can be plotted on a map. 

    :return response.json(): JSON formatted response
    """
    params = {
        'ds': dataset,
        'minLon': bb['min_lon'],
        'minLat': bb['min_lat'],
        'maxLon': bb['max_lon'],
        'maxLat': bb['max_lat'],
        'startTime': start_time.strftime(dt_format),
        'endTime': end_time.strftime(dt_format)
    }

    url = f'{base_url}/timeAvgMapSpark'

    print(f'{url}?{"&".join(f"{k}={v}" for k,v in params.items())}')
    print("Waiting for response from data analysis tool...")
    start = time.perf_counter()
    response = requests.get(url, params=params, verify=False)
    print(f"Time average took {(time.perf_counter() - start):.3f} seconds\n")
    return response.json()
