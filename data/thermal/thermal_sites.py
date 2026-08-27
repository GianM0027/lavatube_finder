"""
Thermal query sites, keyed to landforms rather than to a coordinate grid.

Why this replaces the grid
-------------------------
The first version snapped every crop to a 0.05 degree grid and queried one
window per occupied cell. Two things were wrong with that.

* **The window missed its target.** 0.05 degrees is about 2.96 km at the
  equator, and the extraction window was 3.2 km, so the grid step was very
  nearly the whole window rather than half of it as intended. Measured against
  the crops it was built from, the true landform centre sat a median of 1138 m
  -- 11.4 THEMIS pixels -- from the window centre, and for 10.7% of crops the
  landform fell outside its own window altogether. A THEMIS pixel is ~100 m and
  the median annotated landform is 396 m across, so that is not a rounding
  error, it is pointing at the wrong place.
* **Cells are not landforms.** A cell could hold two unrelated pits, or split
  one pit across a boundary, so a "site" had no physical meaning.

Here a site is a **landform**: the same feature at every resolution replica and
across repeat observations, identified by the same grouping the training split
uses, so a site and a split group are the same object by construction. Its
position is the mean of its members' annotation centres, which are the centres
of the pits themselves, not of the tiles or of the crops.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["MARS_RADIUS_M", "landform_sites"]

#: Mean Mars radius, for converting degrees to metres on the ground.
MARS_RADIUS_M = 3389500.0


def _span_m(latitudes: np.ndarray, longitudes: np.ndarray) -> float:
    """Greatest distance between any two members, in metres."""
    if len(latitudes) < 2:
        return 0.0

    lat = np.radians(latitudes)
    lon = np.radians(longitudes)
    x = MARS_RADIUS_M * lon * np.cos(lat)
    y = MARS_RADIUS_M * lat

    points = np.c_[x, y]
    deltas = points[:, None, :] - points[None, :, :]
    return float(np.hypot(deltas[..., 0], deltas[..., 1]).max())


def landform_sites(
    annotations: pd.DataFrame,
    groups: Sequence[str],
    out_dir: Optional[str] = "data/thermal",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build one thermal query site per landform.

    :param annotations: the annotation table, carrying ``lat``, ``lon_east``,
        ``category_id`` and ``image_name``.
    :param groups: one group key per annotation, from
        ``LandformDataset.group_keys(level="landform")``. Passed in rather than
        recomputed so that a site and a cross-validation group cannot drift
        apart -- there is exactly one implementation of the grouping.
    :param out_dir: where to write ``landform_sites.csv`` and
        ``annotation_sites.csv``. ``None`` writes nothing.
    :return: ``(sites, annotation_sites)``.

    ``sites`` has one row per landform with the position to query, and a
    ``span_m`` column giving how far apart its members are. A large span means
    the grouping merged things a single THEMIS window cannot cover equally well,
    so it is worth checking rather than assuming.
    """
    required = {"lat", "lon_east", "category_id", "image_name"}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"annotations is missing {sorted(missing)}")

    table = annotations.assign(_group=list(groups))

    rows = []
    for group, members in table.groupby("_group", sort=True):
        latitudes = members["lat"].to_numpy(dtype=float)
        longitudes = members["lon_east"].to_numpy(dtype=float)
        finite = np.isfinite(latitudes) & np.isfinite(longitudes)
        if not finite.any():
            continue

        rows.append({
            "_group": group,
            # Mean of the annotation centres: the pits themselves, not the
            # tiles they were cut from and not the crops fed to the network.
            "lat": float(latitudes[finite].mean()),
            "lon_east": float(longitudes[finite].mean()),
            "n_annotations": int(len(members)),
            "span_m": _span_m(latitudes[finite], longitudes[finite]),
            "category_ids": sorted({int(c) for c in members["category_id"]}),
        })

    sites = pd.DataFrame(rows).sort_values(["lat", "lon_east"]).reset_index(drop=True)
    # Prefixed "lf" rather than "site" on purpose: the superseded grid-based
    # pipeline also numbered its cells site_0000 upward, and a stale manifest
    # joined against these ids would match silently and be entirely wrong.
    sites["site_id"] = [f"lf_{i:04d}" for i in range(len(sites))]

    lookup = sites.set_index("_group")["site_id"]
    annotation_sites = pd.DataFrame({
        "image_name": annotations["image_name"].to_numpy(),
        "category_id": annotations["category_id"].to_numpy(),
        "site_id": [lookup.get(g) for g in groups],
    })

    sites = sites.drop(columns="_group")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        sites.to_csv(os.path.join(out_dir, "landform_sites.csv"), index=False)
        annotation_sites.to_csv(
            os.path.join(out_dir, "annotation_sites.csv"), index=False
        )

    if verbose:
        print(f"{len(annotations)} annotations -> {len(sites)} landform sites")
        print(f"  annotations per site : median "
              f"{sites.n_annotations.median():.0f}, max {sites.n_annotations.max()}")
        print(f"  member span (m)      : median {sites.span_m.median():.0f}, "
              f"90th {sites.span_m.quantile(0.9):.0f}, max {sites.span_m.max():.0f}")
        wide = int((sites.span_m > 300).sum())
        print(f"  sites spanning >300 m (3 THEMIS px): {wide}")
        if out_dir:
            print(f"  written to {out_dir}/landform_sites.csv "
                  f"and {out_dir}/annotation_sites.csv")

    return sites, annotation_sites
