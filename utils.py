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


def is_point_in_region(lat, lon, region_coordinates):
    """
    Checks if a given (latitude, longitude) coordinate is inside a defined region.

    :param lat: Float, latitude of the target point.
    :param lon: Float, longitude of the target point.
    :param region_coordinates: List of 4 coordinate pairs defining the region.
    :return: Boolean, True if the point is within the region, False otherwise.
    """
    # 1. Extract latitude bounds
    lats = [pt[0] for pt in region_coordinates]
    min_lat, max_lat = min(lats), max(lats)

    # Check if latitude is out of bounds
    if not (min_lat <= lat <= max_lat):
        return False

    # 2. Extract longitude boundaries based on the polygon's top edge
    # Assumes point 1 is [lat_max, lon_start] and point 2 is [lat_max, lon_end]
    lon_start = region_coordinates[0][1] % 360
    lon_end = region_coordinates[1][1] % 360

    # Normalize the target longitude to the 0-360 range
    lon_norm = lon % 360

    # 3. Check longitude boundary (accounting for wrap-around)
    if lon_start <= lon_end:
        # Standard case (e.g., Arcadia Planitia: 165.9 to 210.4)
        return lon_start <= lon_norm <= lon_end
    else:
        # Wrap-around case (e.g., Acidalia Planitia: 305.12 to 16.18)
        return lon_norm >= lon_start or lon_norm <= lon_end