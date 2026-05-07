import os
import json
import pygmt
import pandas as pd

def region_changed(data_region, region_file):
    if not os.path.exists(region_file):
        return True
    with open(region_file) as f:
        return json.load(f) != data_region

def get_mars_elevation(lat, lon, resolution="01m"):
    """
    Get Mars surface elevation for one or more coordinates.

    Parameters
    ----------
    lat : float or list of float
    lon : float or list of float  (0–360 or -180–180, both work)
    resolution : str, optional
        PyGMT resolution string, default "01m"

    Returns
    -------
    float or list of float : elevation(s) in meters
    """
    lats = [lat] if isinstance(lat, (int, float)) else list(lat)
    lons = [lon] if isinstance(lon, (int, float)) else list(lon)

    points = pd.DataFrame({"longitude": lons, "latitude": lats})

    # Normalize to -180/180 for region calculation
    lons_norm = [x if x <= 180 else x - 360 for x in lons]
    region = [
        min(lons_norm) - 1, max(lons_norm) + 1,
        min(lats) - 1,      max(lats) + 1,
    ]

    grid = pygmt.datasets.load_mars_relief(resolution=resolution, region=region)
    track = pygmt.grdtrack(points=points, grid=grid, newcolname="elevation_m")

    elevations = track["elevation_m"].tolist()
    return elevations[0] if len(elevations) == 1 else elevations