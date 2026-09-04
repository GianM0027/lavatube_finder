"""
Retrieve THEMIS windows for a table of landform sites.

Two phases, both hitting remote archives:

1. **Which observations cover this site.** The ODE ArcGIS footprint layer for
   THEMIS IRPBT4, queried once per site, returning observation ids with
   acquisition time, Mars local solar time and viewing geometry.
2. **The pixels themselves.** ``themis_windows`` opens each product through
   GDAL's ``/vsicurl/`` and fetches only the bytes covering the window, so a
   32x32 patch costs a few kilobytes instead of the 0.4-126 MB the whole product
   would.

Windows are extracted at ``WINDOW_PX`` and any smaller window is a centred
subset of that, so the extraction size only ever needs to be the largest one
under consideration -- narrowing it later is a slice, not another download.

Local solar time is the axis that matters. Cushing et al. (2007) identify a
cave-connected pit by its *diurnal amplitude* being smaller than that of the
collapse pits around it, so a sequence needs frames from different times of day.
Mars Odyssey flies a near sun-synchronous orbit, so a given site is revisited at
only a couple of local times; ``spread_over_local_time`` takes whatever spread
genuinely exists rather than assuming a tidy day/night pair is available.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "WINDOW_PX",
    "find_observations",
    "spread_over_local_time",
    "extract_windows",
    "frame_times",
]

#: Extraction size in THEMIS pixels (~100 m each). A superset: smaller windows
#: are centred slices of this, taken at training time.
WINDOW_PX = 32


def find_observations(
    sites: pd.DataFrame,
    out_path: Optional[str] = "data/thermal/themis_data/observations.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Phase 1: which THEMIS observations cover each site.

    :param sites: rows with ``site_id``, ``lat``, ``lon_east``.
    :return: one row per (site, observation), with local solar time and geometry.
    """
    from tqdm.auto import tqdm

    from data.thermal.themis_measurements import build_records, query_point

    records = []
    failures = 0

    for site in tqdm(
        sites.itertuples(index=False), total=len(sites),
        desc="Phase 1: footprints", disable=not verbose,
    ):
        try:
            payload = query_point(site.lat, site.lon_east)
        except Exception as exc:
            print(f"  {site.site_id}: {type(exc).__name__} {exc}")
            failures += 1
            continue

        for record in build_records(payload, site.lon_east):
            record["site_id"] = site.site_id
            record["site_lat"] = site.lat
            record["site_lon"] = site.lon_east
            records.append(record)

    observations = pd.DataFrame(records)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        observations.to_csv(out_path, index=False)

    if verbose and len(observations):
        covered = observations["site_id"].nunique()
        print(f"{len(observations)} observations over {covered}/{len(sites)} sites"
              f"{f', {failures} query failures' if failures else ''}")
        print(f"  {observations['observation_id'].nunique()} distinct products")

    return observations


def _circular_gap(a: float, b: float, period: float = 24.0) -> float:
    """Shortest distance between two times of day, in hours."""
    diff = abs(a - b) % period
    return min(diff, period - diff)


def spread_over_local_time(group: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Pick ``n`` observations whose local solar times are as far apart as possible.

    Greedy farthest-point selection on the 24 h circle. Targeting fixed clock
    hours would silently return near-identical frames wherever those hours do
    not exist; this extracts whatever diurnal contrast a site genuinely offers
    and degrades gracefully when there is little.

    Rows come back ordered by local solar time, so the temporal model sees a
    coherent progression rather than an arbitrary permutation.
    """
    usable = group.dropna(subset=["mars_lmst_decimal_hours"])
    if len(usable) <= n:
        return usable.sort_values("mars_lmst_decimal_hours")

    hours = usable["mars_lmst_decimal_hours"].tolist()

    picked = [int(np.argmin(hours))]
    while len(picked) < n:
        best = max(
            (i for i in range(len(hours)) if i not in picked),
            key=lambda i: min(_circular_gap(hours[i], hours[j]) for j in picked),
        )
        picked.append(best)

    return usable.iloc[sorted(picked, key=lambda i: hours[i])]


def extract_windows(
    observations: pd.DataFrame,
    sequence_length: int = 3,
    window_px: int = WINDOW_PX,
    out_dir: str = "data/thermal/themis_data/windows",
    manifest_path: str = "data/thermal/themis_data/window_manifest.json",
    verbose: bool = True,
) -> List[dict]:
    """
    Phase 2: fetch the pixels, one ``.npy`` per site.

    Each file holds ``(T, window_px, window_px)`` float32 brightness temperature
    in Kelvin, with 0 marking no data -- the archive's own convention, kept as-is
    so the dataset can build its validity mask from it rather than guessing.

    :return: the manifest, also written to ``manifest_path``.
    """
    from tqdm.auto import tqdm

    from data.thermal.themis_id_to_thermal_value import ensure_index
    from data.thermal.themis_windows import NODATA_KELVIN, read_window_for_observation

    os.makedirs(out_dir, exist_ok=True)
    index_path = ensure_index()

    # Selected explicitly rather than through groupby.apply: pandas' newer
    # ``include_groups=False`` drops the grouping column from the result, and
    # ``site_id`` is needed downstream.
    chosen = {
        site_id: spread_over_local_time(group, sequence_length)
        for site_id, group in observations.groupby("site_id")
    }

    manifest = []
    started = time.time()

    for site_id, group in tqdm(
        chosen.items(), total=len(chosen),
        desc="Phase 2: windows", disable=not verbose,
    ):
        frames, used = [], []

        for observation in group.itertuples():
            try:
                patch = read_window_for_observation(
                    observation.observation_id,
                    observation.site_lat,
                    observation.site_lon,
                    size=window_px,
                    index_path=index_path,
                )
            except Exception as exc:
                print(f"  {site_id}/{observation.observation_id}: "
                      f"{type(exc).__name__} {exc}")
                patch = None

            if patch is None:
                continue

            valid = patch != NODATA_KELVIN
            frames.append(patch)
            used.append({
                "observation_id": observation.observation_id,
                "mars_lmst_decimal_hours": observation.mars_lmst_decimal_hours,
                "valid_fraction": float(valid.mean()),
                "kelvin_median": float(np.median(patch[valid])) if valid.any() else None,
            })

        if not frames:
            continue

        np.save(os.path.join(out_dir, f"{site_id}.npy"),
                np.stack(frames).astype(np.float32))
        manifest.append({
            "site_id": site_id, "n_frames": len(frames),
            "window_px": window_px, "frames": used,
        })

    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)

    if verbose:
        counts = pd.Series([e["n_frames"] for e in manifest])
        print(f"\n{len(manifest)} sites written to {out_dir} "
              f"in {(time.time() - started) / 60:.1f} min")
        print(f"  frames per site: {counts.value_counts().sort_index().to_dict()}")

    return manifest


def frame_times(
    manifest_path: str = "data/thermal/themis_data/window_manifest.json",
    observations_path: str = "data/thermal/themis_data/observations.csv",
) -> dict:
    """
    Mars local solar time and season for every stored frame, keyed by site.

    Why the model needs this
    ------------------------
    ``spread_over_local_time`` uses local solar time to *choose* and *order*
    frames, and then throws it away: what reaches the network is
    ``(T, 2, k, k)`` -- temperature and a validity mask -- with no clock. But
    Cushing et al. (2007) identify a cave-connected pit by a *diurnal
    amplitude*, which is a temperature difference divided by a time difference.
    A model with no time axis cannot compute one, and slot 0 is 03h at one site
    and 07h at the next, so the temporal convolution is not even comparing like
    with like.

    That it matters is measurable in the stored windows. Splitting the
    centre-minus-annulus contrast by local time band, the separation between
    Type-1 and the rest is not merely weaker at some hours -- the *mechanism
    changes sign*:

    ========================  ==========  ==========  ==========
    band                      Type-1      Type-2      Type-4
    ========================  ==========  ==========  ==========
    02-06 h (pre-dawn)        +1.27 K     +0.75 K     -0.27 K
    06-10 h (dawn)            +0.59 K     +0.56 K     -1.09 K
    14-18 h (afternoon)       -0.44 K     -3.53 K     -1.38 K
    18-22 h (dusk)            +0.16 K     -0.51 K     -0.62 K
    ========================  ==========  ==========  ==========

    Pre-dawn is Cushing's regime: a warm, insulated skylight against cold
    ground. The afternoon column is the opposite mechanism -- a large bowl-shaped
    Type-2 pit is 3.5 K *cold* at 14-18 h, which is shadow, i.e. the same
    morphology the optical branch already reads, sampled at 100 m. Averaging the
    two regimes into one sequence without telling the model which is which
    cancels part of the signal.

    Encoding
    --------
    Angles are returned as raw values; ``LandformDataset`` turns them into
    ``sin``/``cos`` pairs so that 23:30 and 00:30 are adjacent rather than a
    24-hour jump.

    Frame order matches the ``(T, k, k)`` order of the stored ``.npy``: both
    come from the same loop in :func:`extract_windows`.

    :param manifest_path: ``window_manifest.json``, which carries local solar
        time per frame.
    :param observations_path: ``observations.csv``, joined on ``observation_id``
        for solar longitude, which the manifest does not record. Missing or
        unmatched, season simply comes back as NaN and the dataset encodes it as
        absent.
    :return: ``{site_id: [(lmst_hours, solar_longitude_deg), ...]}``.
    """
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path) as handle:
        manifest = json.load(handle)

    season = {}
    if os.path.exists(observations_path):
        observations = pd.read_csv(observations_path)
        if {"observation_id", "solar_longitude_deg"}.issubset(observations.columns):
            season = (
                observations.drop_duplicates("observation_id")
                .set_index("observation_id")["solar_longitude_deg"]
                .to_dict()
            )

    times = {}
    for entry in manifest:
        rows = []
        for frame in entry.get("frames", []):
            hour = frame.get("mars_lmst_decimal_hours")
            solar_longitude = season.get(frame.get("observation_id"), np.nan)
            rows.append((
                float(hour) if hour is not None else np.nan,
                float(solar_longitude) if solar_longitude is not None else np.nan,
            ))
        times[entry["site_id"]] = rows

    return times
