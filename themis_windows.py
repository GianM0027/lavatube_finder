"""
Read small THEMIS windows straight from the archive, without downloading products.

Phase 2 (``themis_id_to_thermal_value.py``) fetches whole PBT products so it can
decode them locally. That is very expensive for this pipeline: products run
0.4-126 MB (median 22 MB), and the model only needs a 32x32 patch -- about 4 KB
-- centred on each site. Downloading the 312 products behind a T=3 sequence
costs roughly 7 GB of cache plus 7 GB of decoded ``.npy``.

The ASU archive supports HTTP range requests, so GDAL can open a product through
``/vsicurl/`` and fetch only the bytes covering the requested window. That turns
~14 GB of traffic and disk into a few megabytes.

Values are brightness temperature in Kelvin, with 0 marking no data.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

# Keep GDAL from listing remote directories on open, which wastes round trips.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".IMG,.xml,.LBL,.lbl")

import rasterio
from rasterio.windows import Window
from pyproj import CRS, Transformer

__all__ = ["read_window_at", "read_window_for_observation", "NODATA_KELVIN"]

#: PBT products use 0 for "no data", not a negative sentinel.
NODATA_KELVIN = 0.0


def _to_product_xy(dataset, lat: float, lon_east: float) -> Tuple[float, float]:
    """Project a site's lat/lon into the product's own CRS."""
    radius = dataset.crs.to_dict().get("R") or dataset.crs.ellipsoid.semi_major_metre
    geographic = CRS.from_proj4(f"+proj=longlat +R={radius} +no_defs")
    transformer = Transformer.from_crs(geographic, dataset.crs, always_xy=True)
    # Products are centred near their own longitude; feeding -180..180 keeps the
    # sinusoidal projection on the correct side of its central meridian.
    lon = ((lon_east + 180.0) % 360.0) - 180.0
    return transformer.transform(lon, lat)


def read_window_at(
    product_url: str,
    lat: float,
    lon_east: float,
    size: int = 32,
) -> Optional[np.ndarray]:
    """
    Fetch a ``size`` x ``size`` window centred on a site, in Kelvin.

    Only the bytes covering the window are transferred. Windows that fall
    partly outside the product are zero-padded, matching the archive's own
    no-data convention; windows falling entirely outside return None.

    :param product_url: https URL of the PBT label (.xml) or image (.IMG).
    :param lat: Site latitude in degrees.
    :param lon_east: Site east longitude in degrees (0..360 or -180..180).
    :param size: Side length of the window in THEMIS pixels (~100 m each).
    :return: (size, size) float32 array of brightness temperature, or None.
    """
    target = product_url if product_url.startswith("/vsicurl/") else f"/vsicurl/{product_url}"

    with rasterio.open(target) as dataset:
        x, y = _to_product_xy(dataset, lat, lon_east)
        row, col = dataset.index(x, y)

        half = size // 2
        row_off, col_off = row - half, col - half

        # Reject sites that miss the product entirely.
        if row_off + size <= 0 or col_off + size <= 0:
            return None
        if row_off >= dataset.height or col_off >= dataset.width:
            return None

        # boundless=True lets rasterio pad the parts that fall outside.
        patch = dataset.read(
            1,
            window=Window(col_off, row_off, size, size),
            boundless=True,
            fill_value=NODATA_KELVIN,
        )

    return patch.astype(np.float32)


def read_window_for_observation(
    observation_id: str,
    lat: float,
    lon_east: float,
    size: int = 32,
    index_path=None,
) -> Optional[np.ndarray]:
    """
    Same as :func:`read_window_at`, but resolves the product URL from an
    observation ID via the ASU index.

    The index (~42 MB) is downloaded once and cached by
    ``themis_id_to_thermal_value.ensure_index``; pass ``index_path`` to reuse a
    resolved path across many calls instead of re-checking it each time.
    """
    from themis_id_to_thermal_value import (
        build_urls,
        ensure_index,
        find_product,
        normalize_id,
    )

    if index_path is None:
        index_path = ensure_index()

    found = find_product(index_path, normalize_id(observation_id))
    if not found:
        return None

    image_url, label_urls = build_urls(found)

    # PDS4 (.xml) carries the georeferencing GDAL needs; try the labels first
    # and fall back to the raw image.
    for url in list(label_urls) + [image_url]:
        try:
            return read_window_at(url, lat, lon_east, size)
        except rasterio.errors.RasterioIOError:
            continue

    return None
