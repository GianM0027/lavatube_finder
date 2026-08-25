"""
Geographic coordinates for the training images, for THEMIS thermal retrieval.

The crops in data/ come from two sources whose georeferencing differs, and one
of them is subtly malformed, so the CRS cannot simply be trusted as written:

* DeepLandforms crops carry a correct Equirectangular CRS.
* Plain-terrain crops inherit their CRS from the source HiRISE products. Most
  are Equirectangular with the standard parallel mistakenly stored in
  ``latitude_of_origin`` while ``standard_parallel_1`` is 0 -- a known quirk of
  planetary PDS/ISIS products. Read literally, this offsets every latitude by
  exactly the standard parallel (verified: 35 deg, 5 deg, 45 deg, 15 deg errors)
  and mis-scales longitude. A few products are Polar Stereographic instead and
  are correct as written.

`center_latlon` detects the malformed pattern and rebuilds the projection with
the standard parallel in the right slot, leaving every other CRS untouched.
Validated against the LBL-derived lat/lon recorded in
data/plain_terrain_dataset/plain_terrain_annotations.csv.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Tuple

import rasterio
from pyproj import CRS, Transformer

__all__ = ["center_latlon", "ground_extent_m", "THEMIS_GSD_M"]

# Nominal THEMIS IR ground sampling distance (metres per pixel).
THEMIS_GSD_M = 100.0


def _param(wkt: str, name: str) -> float:
    """Read a PROJECTION PARAMETER out of a WKT string, defaulting to 0."""
    match = re.search(rf'PARAMETER\["{name}",([-0-9.eE+]+)\]', wkt)
    return float(match.group(1)) if match else 0.0


def _projection_name(wkt: str) -> str:
    match = re.search(r'PROJECTION\["([^"]+)"\]', wkt)
    return match.group(1) if match else ""


def _sphere_radius(crs: CRS) -> float:
    """
    Mars sphere radius in metres for this CRS.

    Read straight off the ellipsoid rather than via ``crs.to_dict()``: that
    helper round-trips the CRS through a PROJ string, which emits a
    "you will likely lose important projection information" warning and is
    lossy in general. Every CRS here defines a perfect sphere (inverse
    flattening 0), so the semi-major axis is the radius.
    """
    return crs.ellipsoid.semi_major_metre


@lru_cache(maxsize=None)
def _transformer_for(wkt: str) -> Transformer:
    """
    Build a projected -> geographic transformer, correcting the malformed
    Equirectangular pattern described in the module docstring.
    """
    crs = CRS.from_wkt(wkt)
    radius = _sphere_radius(crs)

    lat_origin = _param(wkt, "latitude_of_origin")
    std_par_1 = _param(wkt, "standard_parallel_1")

    is_equirect = "equirectangular" in _projection_name(wkt).lower()
    malformed = is_equirect and lat_origin != 0.0 and std_par_1 == 0.0

    if malformed:
        # The standard parallel was written into latitude_of_origin. Put it back.
        crs = CRS.from_proj4(
            f"+proj=eqc +R={radius} +lat_ts={lat_origin} +lat_0=0 "
            f"+lon_0={_param(wkt, 'central_meridian')} +x_0=0 +y_0=0 "
            f"+units=m +no_defs"
        )

    geographic = CRS.from_proj4(f"+proj=longlat +R={radius} +no_defs")
    return Transformer.from_crs(crs, geographic, always_xy=True)


def center_latlon(image_path: str) -> Tuple[float, float]:
    """
    Centre of a georeferenced crop as (latitude, east longitude in 0..360).

    :param image_path: Path to a georeferenced GeoTIFF crop.
    :return: (lat, lon_east) in degrees.
    """
    with rasterio.open(image_path) as src:
        bounds = src.bounds
        x = (bounds.left + bounds.right) / 2.0
        y = (bounds.bottom + bounds.top) / 2.0
        transformer = _transformer_for(src.crs.to_wkt())

    lon, lat = transformer.transform(x, y)
    return lat, lon % 360.0


def ground_extent_m(image_path: str) -> Tuple[float, float]:
    """
    Ground extent of a crop in metres as (width, height).

    Useful for judging how many THEMIS samples a crop actually spans:
    ``width / THEMIS_GSD_M``.
    """
    with rasterio.open(image_path) as src:
        bounds = src.bounds
        return abs(bounds.right - bounds.left), abs(bounds.top - bounds.bottom)
