#!/usr/bin/env python3
"""
THEMIS Phase 2 - single PBT product -> matrix

Input:
    --id I88015002

Pipeline:
    observation ID
      -> official ASU ODTGEO master index (cached once)
      -> exact PBT path
      -> download IMG + detached label
      -> decode matrix
      -> save .npy + metadata JSON

No ODE calls are used here.

Install:
    pip install requests numpy rasterio

Run:
    python themis_phase2_single_pbt.py --id I88015002
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import requests

ASU_ROOT = "https://static.mars.asu.edu/pds/ODTGEO_v2/"
INDEX_URL = ASU_ROOT + "index/CMIDX_ODTIP.TAB"

# Anchored to this file rather than the working directory, so the index is found
# whether the caller runs the notebook in data/thermal/, imports the module from
# the repo root, or invokes this script directly.
_HERE = Path(__file__).resolve().parent

CACHE_DIR = _HERE / "themis_cache"
INDEX_FILE = CACHE_DIR / "CMIDX_ODTIP.TAB"

# Whole-product downloads land here. The pipeline no longer needs them --
# themis_windows.py fetches just the bytes covering each window over HTTP range
# requests -- but the path is kept for direct CLI use of this script.
OUT_DIR = _HERE / "themis_products"

HEADERS = {
    "User-Agent": "THEMIS-lavatube-phase2/1.0",
    "Accept": "*/*",
}


def session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def download(url: str, dest: Path, timeout=(10, 120), retries=3):
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    s = session()
    last = None

    for attempt in range(retries):
        try:
            print(f"Downloading: {url}")
            with s.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
                return dest
        except Exception as e:
            last = e
            print(f"  attempt {attempt+1}/{retries} failed: {e}")
            if attempt + 1 < retries:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Could not download {url}\n{last}")


#: A THEMIS observation id is ``I`` plus eight characters. In the primary
#: mission those eight are all digits (``I88015002``). Once the orbit counter
#: passed 99999 the leading digit was replaced by a letter, so extended-mission
#: products read ``IA1255015``, ``IB…`` and so on. Matching only ``I\d{8}``
#: silently rejected every extended-mission product: 161 of the 508 observations
#: covering these sites, 32% of the archive available here.
_OBSERVATION_ID = re.compile(r"I[0-9A-Z]\d{7}", re.I)

#: The image filename carrying such an id, as written in the ASU index.
_IMG_NAME = re.compile(r"I[0-9A-Z]\d{7}PBT\.IMG", re.I)


def normalize_id(obs_id: str) -> str:
    """
    Extract the canonical observation id from a product name.

    Accepts the id on its own or with an archive suffix (``IA1255015PBT``),
    in either case returning the nine-character id in upper case.
    """
    m = _OBSERVATION_ID.search(obs_id.strip())
    if not m:
        raise ValueError(
            f"{obs_id!r} does not contain a THEMIS observation ID "
            "such as I88015002 or IA1255015"
        )
    return m.group(0).upper()


def ensure_index() -> Path:
    if INDEX_FILE.exists() and INDEX_FILE.stat().st_size > 1_000_000:
        print(f"Using cached ASU index: {INDEX_FILE}")
        return INDEX_FILE

    print("Downloading official THEMIS projected-product index once...")
    return download(INDEX_URL, INDEX_FILE, timeout=(15, 180))


def find_product(index_path: Path, obs_id: str):
    """
    Search the ASU ODTGEO cumulative index for the exact observation and PBT type.
    The index is CSV-like fixed records.
    """
    with open(index_path, "r", encoding="latin-1", errors="ignore", newline="") as f:
        for line_no, line in enumerate(f, 1):
            if obs_id not in line.upper():
                continue

            try:
                row = next(csv.reader([line]))
            except Exception:
                continue

            vals = [v.strip().strip('"') for v in row]

            # We don't hard-code column count; search fields.
            upper = [v.upper() for v in vals]

            # Exact observation and a PBT indicator must both be present.
            if obs_id not in upper:
                continue

            if not any(v == "PBT" or "PBT" in v for v in upper):
                continue

            # Find field that looks like IMG filename. The id shape has to match
            # _OBSERVATION_ID, extended-mission products included -- hard-coding
            # eight digits here silently dropped every ``IA…PBT.IMG`` even once
            # the id itself had been accepted.
            img_name = next(
                (v for v in vals if _IMG_NAME.fullmatch(v)),
                None,
            )

            # Find field that contains a data directory.
            path_field = next(
                (
                    v for v in vals
                    if "/" in v and ("PBT" in v.upper() or "ODTIP" in v.upper())
                ),
                None,
            )

            # Common index layout has path in the field immediately before file.
            if img_name:
                idx = vals.index(img_name)
                if idx > 0 and "/" in vals[idx - 1]:
                    path_field = vals[idx - 1]

            if img_name:
                return {
                    "line_number": line_no,
                    "row": vals,
                    "img_name": img_name,
                    "path": path_field,
                    "raw": line.strip(),
                }

    return None


def build_urls(found):
    path = (found["path"] or "").replace("\\", "/").strip("/")

    # Static archive directory names are normally lowercase.
    path_lower = "/".join(part.lower() for part in path.split("/"))

    img_name = found["img_name"]
    base = ASU_ROOT + (path_lower + "/" if path_lower else "")

    img_url = base + img_name

    stem = img_name[:-4]
    labels = [
        base + stem + ".LBL",
        base + stem + ".lbl",
        base + stem + ".xml",
        base + stem + ".XML",
    ]

    return img_url, labels


def download_first_existing(urls, dest_dir):
    s = session()
    for url in urls:
        try:
            r = s.get(url, timeout=(10, 30), stream=True)
            if r.status_code == 404:
                r.close()
                continue
            r.raise_for_status()

            name = url.rsplit("/", 1)[-1]
            dest = dest_dir / name
            tmp = dest.with_suffix(dest.suffix + ".part")
            dest.parent.mkdir(parents=True, exist_ok=True)

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024 * 512):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
            return dest, url
        except requests.RequestException:
            continue

    return None, None


def parse_pds3_label(text: str) -> dict:
    """
    Minimal PDS3 label parser for common IMAGE fields.
    """
    def get(name):
        # Handles NAME = value, quoted or unquoted.
        m = re.search(
            rf"(?mi)^\s*{re.escape(name)}\s*=\s*(.+?)\s*$",
            text,
        )
        if not m:
            return None
        v = m.group(1).strip()
        if "/*" in v:
            v = v.split("/*", 1)[0].strip()
        return v.strip('"')

    fields = {}
    for key in [
        "LINES",
        "LINE_SAMPLES",
        "SAMPLE_BITS",
        "SAMPLE_TYPE",
        "SCALING_FACTOR",
        "OFFSET",
        "CORE_MULTIPLIER",
        "CORE_BASE",
        "MISSING_CONSTANT",
        "INVALID_CONSTANT",
        "MAP_SCALE",
        "CENTER_LATITUDE",
        "CENTER_LONGITUDE",
        "LINE_PROJECTION_OFFSET",
        "SAMPLE_PROJECTION_OFFSET",
        "MAP_PROJECTION_TYPE",
    ]:
        fields[key] = get(key)

    # Find byte offset of detached/attached IMAGE if present.
    fields["IMAGE_POINTER"] = get("^IMAGE")
    return fields


def dtype_from_label(sample_type, bits):
    if not sample_type or not bits:
        return None

    bits = int(re.search(r"\d+", str(bits)).group(0))
    nbytes = bits // 8
    st = sample_type.upper()

    if "REAL" in st or "FLOAT" in st:
        if nbytes == 4:
            return np.dtype(">f4" if ("MSB" in st or "IEEE_REAL" in st) else "<f4")
        if nbytes == 8:
            return np.dtype(">f8" if "MSB" in st else "<f8")

    unsigned = "UNSIGNED" in st
    endian = ">" if ("MSB" in st or "SUN" in st) else "<"
    kind = "u" if unsigned else "i"

    return np.dtype(f"{endian}{kind}{nbytes}")


def parse_number(v, default=None):
    if v is None:
        return default
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", str(v))
    return float(m.group(0)) if m else default


def read_with_rasterio(img_path: Path, label_path: Path | None):
    try:
        import rasterio
    except Exception:
        return None, None

    candidates = [label_path, img_path] if label_path else [img_path]

    for p in candidates:
        if not p:
            continue
        try:
            with rasterio.open(p) as ds:
                arr = ds.read(1).astype(np.float32)
                return arr, {
                    "reader": "rasterio/GDAL",
                    "driver": ds.driver,
                    "width": ds.width,
                    "height": ds.height,
                    "count": ds.count,
                    "dtype": str(ds.dtypes[0]),
                    "nodata": ds.nodata,
                    "crs": str(ds.crs) if ds.crs else None,
                    "transform": list(ds.transform) if ds.transform else None,
                }
        except Exception:
            pass

    return None, None


def read_pds3_binary(img_path: Path, label_text: str):
    meta = parse_pds3_label(label_text)

    lines = parse_number(meta["LINES"])
    samples = parse_number(meta["LINE_SAMPLES"])
    bits = parse_number(meta["SAMPLE_BITS"])

    if lines is None or samples is None or bits is None:
        raise RuntimeError(
            "Label found, but LINES/LINE_SAMPLES/SAMPLE_BITS could not be parsed."
        )

    lines = int(lines)
    samples = int(samples)

    dtype = dtype_from_label(meta["SAMPLE_TYPE"], int(bits))
    if dtype is None:
        raise RuntimeError(f"Unsupported SAMPLE_TYPE: {meta['SAMPLE_TYPE']}")

    # Detached PDS3 IMG usually begins at byte 0.
    offset_bytes = 0

    ptr = meta.get("IMAGE_POINTER")
    if ptr:
        # If pointer is an explicit byte value such as 1 <BYTES>.
        if "<BYTES>" in ptr.upper():
            n = parse_number(ptr)
            if n is not None:
                # PDS pointers are 1-based.
                offset_bytes = max(0, int(n) - 1)

    count = lines * samples
    arr = np.fromfile(
        img_path,
        dtype=dtype,
        count=count,
        offset=offset_bytes,
    )

    if arr.size != count:
        raise RuntimeError(
            f"Binary size mismatch: expected {count} pixels, read {arr.size}."
        )

    arr = arr.reshape(lines, samples).astype(np.float32)

    # PDS scaling conventions if present.
    multiplier = (
        parse_number(meta.get("SCALING_FACTOR"))
        if meta.get("SCALING_FACTOR") is not None
        else parse_number(meta.get("CORE_MULTIPLIER"), 1.0)
    )
    if multiplier is None:
        multiplier = 1.0

    base = (
        parse_number(meta.get("OFFSET"))
        if meta.get("OFFSET") is not None
        else parse_number(meta.get("CORE_BASE"), 0.0)
    )
    if base is None:
        base = 0.0

    arr = arr * float(multiplier) + float(base)

    return arr, {
        "reader": "manual PDS3",
        "lines": lines,
        "samples": samples,
        "sample_bits": int(bits),
        "sample_type": meta["SAMPLE_TYPE"],
        "scaling_factor": multiplier,
        "offset": base,
        "label_fields": meta,
    }


def summarize(arr):
    finite = arr[np.isfinite(arr)]
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "valid_pixels": int(finite.size),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    args = ap.parse_args()

    obs_id = normalize_id(args.id)

    print("=" * 70)
    print("THEMIS Phase 2 - single PBT product -> matrix")
    print("=" * 70)
    print(f"Observation ID: {obs_id}")
    print()

    index_path = ensure_index()

    print("Searching official ASU ODTGEO index locally...")
    found = find_product(index_path, obs_id)

    if not found:
        raise RuntimeError(
            f"{obs_id} was not found as a PBT product in the ASU ODTGEO index."
        )

    print(f"Found on index line {found['line_number']}")
    print(f"File: {found['img_name']}")
    print(f"Path: {found['path']}")

    img_url, label_urls = build_urls(found)

    pdir = CACHE_DIR / obs_id
    img_path = pdir / found["img_name"]

    download(img_url, img_path, timeout=(15, 180))

    label_path, label_url = download_first_existing(label_urls, pdir)

    if label_path:
        print(f"Label: {label_path.name}")
    else:
        print("No detached .LBL/.xml label found by conventional name.")

    arr = None
    read_meta = None

    # First use GDAL/rasterio because it handles PDS map-projection metadata.
    arr, read_meta = read_with_rasterio(img_path, label_path)

    # Fallback: manual PDS3 binary decoding.
    if arr is None and label_path and label_path.suffix.lower() == ".lbl":
        text = label_path.read_text(encoding="latin-1", errors="ignore")
        arr, read_meta = read_pds3_binary(img_path, text)

    if arr is None:
        raise RuntimeError(
            "The IMG and label were downloaded successfully, but the raster "
            "could not yet be decoded. Send the console output and the label "
            "file name; the data-access part is nevertheless solved."
        )

    OUT_DIR.mkdir(exist_ok=True)

    matrix_path = OUT_DIR / f"{obs_id}_PBT_matrix.npy"
    np.save(matrix_path, arr)

    info = {
        "observation_id": obs_id,
        "img_url": img_url,
        "label_url": label_url,
        "index_path": found["path"],
        "index_file": found["img_name"],
        "read_metadata": read_meta,
        "array": summarize(arr),
    }

    report_path = OUT_DIR / f"{obs_id}_report.json"
    report_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    s = info["array"]

    print()
    print("SUCCESS")
    print(f"Shape : {tuple(s['shape'])}")
    print(f"Min   : {s['min']}")
    print(f"Max   : {s['max']}")
    print(f"Mean  : {s['mean']}")
    print(f"Matrix: {matrix_path}")
    print(f"Report: {report_path}")
    print()
    print(
        "Important: this script preserves the physical values decoded from the "
        "PBT product. We will verify the label units/scaling before treating "
        "the printed range as Kelvin in the final pipeline."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
