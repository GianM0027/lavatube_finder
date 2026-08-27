#!/usr/bin/env python3
"""
THEMIS Phase 1 - ArcGIS footprint query
=======================================

Given Mars coordinates, query the official ODE ArcGIS footprint layer for
THEMIS IRPBT4 products and report observation IDs + acquisition metadata.

Important:
- User longitude may be 0..360 east (e.g. 241.3 E).
- The c0a ArcGIS layer uses -180..180 longitudes, so 241.3 E -> -118.7.

Output:
- console table
- CSV
- raw JSON response for debugging/reproducibility

Example:
    python themis_phase1_arcgis.py --lat -1.686 --lon 241.3

Dependency:
    pip install requests
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import requests

LAYER_QUERY_URL = (
    "https://mars1ms.rsl.wustl.edu/arcgis/rest/services/"
    "mars_ody_themis_irpbt4/c0a/MapServer/0/query"
)

OUT_DIR = Path("data/themis_phase1_output")

FIELDS = [
    "ProductId",
    "UTCstart",
    "UTCend",
    "EmAngle",
    "InAngle",
    "PhAngle",
    "SolLong",
    "ProdVer",
    "ProductLid",
    "ODEId",
]


def lon_360(lon: float) -> float:
    return lon % 360.0


def lon_180(lon: float) -> float:
    x = ((lon + 180.0) % 360.0) - 180.0
    # avoid -180/180 ambiguity
    return 180.0 if abs(x + 180.0) < 1e-12 else x


def parse_utc(s: str | None) -> datetime | None:
    if not s:
        return None

    s = s.strip()
    if not s:
        return None

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def julian_date(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    y = dt.year
    m = dt.month
    d = (
        dt.day
        + dt.hour / 24.0
        + dt.minute / 1440.0
        + (dt.second + dt.microsecond / 1e6) / 86400.0
    )

    if m <= 2:
        y -= 1
        m += 12

    a = math.floor(y / 100)
    b = 2 - a + math.floor(a / 4)

    return (
        math.floor(365.25 * (y + 4716))
        + math.floor(30.6001 * (m + 1))
        + d + b - 1524.5
    )


def mars_lmst_hours(dt: datetime, east_lon_deg: float) -> float:
    """
    Approximate Local Mean Solar Time on Mars.
    Accuracy is more than adequate for selecting observations within
    hour-scale evening / early-morning windows.
    """
    jd_utc = julian_date(dt)

    # TT - UTC = 69.184 s for these modern observations.
    jd_tt = jd_utc + 69.184 / 86400.0

    msd = (jd_tt - 2405522.0028779) / 1.0274912517
    mtc = (24.0 * ((msd + 0.00096) % 1.0)) % 24.0

    return (mtc + lon_360(east_lon_deg) / 15.0) % 24.0


def hhmm(hours: float | None) -> str:
    if hours is None:
        return ""

    total_minutes = int(round((hours % 24.0) * 60.0)) % (24 * 60)
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h:02d}:{m:02d}"


def query_point(lat: float, lon_user: float, timeout=45) -> dict:
    lon_layer = lon_180(lon_user)

    params = {
        "f": "json",
        "where": "1=1",
        "geometry": f"{lon_layer:.10f},{lat:.10f}",
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": ",".join(FIELDS),
        "returnGeometry": "false",
        "returnIdsOnly": "false",
    }

    headers = {
        "User-Agent": "THEMIS-lavatube-phase1/1.0",
        "Accept": "application/json,*/*",
    }

    r = requests.get(
        LAYER_QUERY_URL,
        params=params,
        headers=headers,
        timeout=(10, timeout),
    )
    r.raise_for_status()

    payload = r.json()

    if "error" in payload:
        raise RuntimeError(
            "ArcGIS returned an error:\n"
            + json.dumps(payload["error"], indent=2)
        )

    return payload


def clean_product_id(pid: str | None) -> str:
    """
    Strip the archive suffix from an IRPBT product id.

    An observation id is ``I`` plus eight characters. In the primary mission
    those eight are digits (``I88015002PBT``); once the orbit counter passed
    99999 the leading digit became a letter, so extended-mission products read
    ``IA1255015PBT``. Matching only ``I\\d{8}`` left those ids with their
    ``PBT`` suffix attached, which every downstream lookup then failed on.
    """
    if not pid:
        return ""

    pid = pid.strip().upper()

    m = re.match(r"(I[0-9A-Z]\d{7})(?:PBT)?", pid)
    return m.group(1) if m else pid


def build_records(payload: dict, lon_user: float) -> list[dict]:
    records = []

    for feature in payload.get("features", []):
        a = feature.get("attributes") or {}

        utc_raw = a.get("UTCstart")
        dt = parse_utc(utc_raw)

        lmst = mars_lmst_hours(dt, lon_user) if dt else None

        records.append({
            "observation_id": clean_product_id(a.get("ProductId")),
            "product_id_raw": a.get("ProductId") or "",
            "utc_start": utc_raw or "",
            "utc_end": a.get("UTCend") or "",
            "mars_lmst_decimal_hours": round(lmst, 4) if lmst is not None else "",
            "mars_lmst_hhmm": hhmm(lmst),
            "solar_longitude_deg": a.get("SolLong"),
            "emission_angle_deg": a.get("EmAngle"),
            "incidence_angle_deg": a.get("InAngle"),
            "phase_angle_deg": a.get("PhAngle"),
            "product_version": a.get("ProdVer") or "",
            "product_lid": a.get("ProductLid") or "",
            "ode_id": a.get("ODEId") or "",
        })

    records.sort(key=lambda x: (x["utc_start"], x["observation_id"]))
    return records


def write_csv(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Find official THEMIS IRPBT4 footprints intersecting a Mars point "
            "and report their acquisition times."
        )
    )
    ap.add_argument("--lat", required=True, type=float)
    ap.add_argument("--lon", required=True, type=float,
                    help="East longitude; both 0..360 and -180..180 accepted.")
    args = ap.parse_args()

    if not -90.0 <= args.lat <= 90.0:
        raise ValueError("Latitude must be between -90 and +90 degrees.")

    l360 = lon_360(args.lon)
    l180 = lon_180(args.lon)

    print("=" * 76)
    print("THEMIS Phase 1 - official IRPBT4 ArcGIS footprint query")
    print("=" * 76)
    print(f"Input latitude       : {args.lat:.6f}")
    print(f"Input longitude east : {l360:.6f} deg E")
    print(f"ArcGIS longitude     : {l180:.6f} deg (-180..180)")
    print()
    print("Querying official ODE/PDS ArcGIS IRPBT4 layer...")

    payload = query_point(args.lat, args.lon)

    OUT_DIR.mkdir(exist_ok=True)

    raw_path = OUT_DIR / "raw_arcgis_response.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    records = build_records(payload, args.lon)

    print(f"Features returned: {len(payload.get('features', []))}")
    print(f"Observations parsed: {len(records)}")
    print()

    if not records:
        print("No IRPBT4 footprint intersects this exact point.")
        print(f"Raw server response saved to: {raw_path}")
        print(
            "If this happens despite known nearby products, the next diagnostic "
            "will inspect their footprints directly rather than changing APIs."
        )
        return

    header = (
        f"{'Observation':<13} "
        f"{'UTC start':<28} "
        f"{'Mars LMST':<10} "
        f"{'Ls':>8} "
        f"{'Emission':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in records:
        ls = (
            f"{float(r['solar_longitude_deg']):.3f}"
            if r["solar_longitude_deg"] is not None
            else "-"
        )
        em = (
            f"{float(r['emission_angle_deg']):.3f}"
            if r["emission_angle_deg"] is not None
            else "-"
        )

        print(
            f"{r['observation_id']:<13} "
            f"{(r['utc_start'] or '-'):<28} "
            f"{(r['mars_lmst_hhmm'] or '-'):<10} "
            f"{ls:>8} "
            f"{em:>10}"
        )

    safe_lat = f"{args.lat:.5f}".replace("-", "m").replace(".", "p")
    safe_lon = f"{l360:.5f}".replace(".", "p")
    csv_path = OUT_DIR / f"observations_lat_{safe_lat}_lon_{safe_lon}.csv"

    write_csv(records, csv_path)

    print()
    print(f"CSV saved : {csv_path}")
    print(f"Raw JSON  : {raw_path}")
    print()
    print("Phase 1 complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
