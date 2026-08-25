"""
Cut matched positive/negative crop pairs from the DeepLandforms tiles.

Why this exists
---------------
The original negative class (``data/plain_terrain_dataset``) was collected by a
different process, from different HiRISE products, at a different resolution and
a different ground footprint than the labelled landforms. That let a classifier
separate it from the pit classes without looking at morphology at all:

* resolution alone            -> 52.5% (4-class)
* ground extent alone         -> 88.5% (plain vs rest)
* all image statistics        -> 97.9% (plain vs rest)

Fixing those one at a time does not work -- normalising resolution simply hands
the job to extent (measured: 66.3% -> 65.6%). They are symptoms of one root
cause, which is that positives and negatives do not share provenance.

What this does
--------------
For every DeepLandforms tile, cut two crops of *identical size*:

* a **positive** centred on the annotated landform, and
* a **negative** from a region of the same tile that does not overlap the
  annotation.

Both come from the same product, the same ISIS pipeline, the same resolution
band, the same dtype and the same illumination, and -- because the pair is cut
to one size -- the same ground footprint. Every acquisition confound cancels
within the pair.

Caveat: annotators marked only their target landform, so a negative crop may
contain an unlabelled one. That is label noise, and it is far less damaging than
a confound that lets a model score 97.9% without learning anything.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

__all__ = ["plan_pair", "cut_pairs"]

#: Smallest crop worth keeping, in pixels.
MIN_SIDE_PX = 200


def plan_pair(
    width: int, height: int, bbox, margin: float = 1.2
) -> Optional[Tuple[dict, dict, int]]:
    """
    Work out where a matched positive/negative pair can sit inside one tile.

    The negative goes in the widest strip of tile that the annotation leaves
    free; the positive is centred on the annotation. Both are squares of the
    same side, so the pair carries no size difference for a model to exploit.

    The side is taken as large as the free strip allows, and the pair is
    rejected unless that is still wide enough to hold the whole annotation with
    ``margin`` to spare. Sizing the crop to the *gap* alone is not enough: on
    tiles where the landform is larger than the free strip it produces a
    positive zoomed inside the pit -- a black or featureless square with none of
    the rim and shadow that distinguish the classes.

    :return: (positive_window, negative_window, side) or None if there is no
             room for a usable pair.
    """
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None

    box_x, box_y, box_w, box_h = (float(v) for v in bbox)

    # Free space on each side of the annotation.
    gaps = {
        "left": box_x,
        "right": width - (box_x + box_w),
        "top": box_y,
        "bottom": height - (box_y + box_h),
    }
    side_name = max(gaps, key=gaps.get)

    # As large as the tile and the free strip permit ...
    side = int(min(gaps[side_name], width, height))
    # ... but only if the landform still fits inside it with margin.
    required = max(max(box_w, box_h) * margin, MIN_SIDE_PX)

    if side < required:
        return None

    # Negative: inside the free strip, centred on the tile's other axis.
    if side_name == "left":
        neg_col, neg_row = 0, (height - side) / 2
    elif side_name == "right":
        neg_col, neg_row = width - side, (height - side) / 2
    elif side_name == "top":
        neg_col, neg_row = (width - side) / 2, 0
    else:
        neg_col, neg_row = (width - side) / 2, height - side

    # Positive: centred on the annotation, nudged to stay inside the tile.
    pos_col = min(max(box_x + box_w / 2 - side / 2, 0), width - side)
    pos_row = min(max(box_y + box_h / 2 - side / 2, 0), height - side)

    positive = {"col_off": int(pos_col), "row_off": int(pos_row)}
    negative = {"col_off": int(neg_col), "row_off": int(neg_row)}
    return positive, negative, side


def _write_crop(source, window: Window, destination: str) -> None:
    """Write one window out, carrying its georeferencing across."""
    data = source.read(1, window=window)
    profile = source.profile.copy()
    profile.update(
        width=int(window.width),
        height=int(window.height),
        transform=source.window_transform(window),
    )
    with rasterio.open(destination, "w", **profile) as sink:
        sink.write(data, 1)


def cut_pairs(
    annotations: pd.DataFrame,
    out_dir: str = "data/matched_crops",
    negative_category_id: int = 3,
    overlap_fraction: float = 0.0,
    margin: float = 1.2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Cut a matched pair from every DeepLandforms tile that has room for one.

    :param annotations: rows with img_path / image_name / bbox / category_id.
    :param out_dir: directory for the generated crops.
    :param negative_category_id: label to give the negative crops.
    :param overlap_fraction: maximum tolerated overlap between the negative
        window and the annotation, as a fraction of the negative's area.
        Zero keeps only strictly disjoint negatives.
    :return: annotation rows describing the generated crops.
    """
    os.makedirs(out_dir, exist_ok=True)

    records = []
    skipped_no_room = 0
    skipped_overlap = 0

    # A tile can carry more than one annotation (1846 rows over 1836 tiles), and
    # they are not always the same class. Work per tile so that the output names
    # cannot collide, and so the negative can be checked against *every*
    # annotation on the tile rather than only the one it was cut for.
    for (img_path, image_name), tile_rows in annotations.groupby(
        ["img_path", "image_name"], sort=False
    ):
        row = tile_rows.iloc[0]
        all_boxes = [
            b for b in tile_rows["bbox"]
            if isinstance(b, (list, tuple)) and len(b) == 4
        ]
        path = os.path.join(img_path, image_name)

        try:
            source = rasterio.open(path)
        except rasterio.errors.RasterioIOError:
            continue

        with source:
            plan = plan_pair(source.width, source.height, row["bbox"], margin=margin)
            if plan is None:
                skipped_no_room += 1
                continue

            positive, negative, side = plan

            # The negative must clear every annotation on this tile, not just
            # the one the positive was cut for.
            clipped = False
            for box_x, box_y, box_w, box_h in ((float(v) for v in b) for b in all_boxes):
                ox = max(0, min(negative["col_off"] + side, box_x + box_w) - max(negative["col_off"], box_x))
                oy = max(0, min(negative["row_off"] + side, box_y + box_h) - max(negative["row_off"], box_y))
                if (ox * oy) > overlap_fraction * side * side:
                    clipped = True
                    break

            if clipped:
                skipped_overlap += 1
                continue

            stem, suffix = os.path.splitext(image_name)

            for kind, window_spec, category in (
                ("pos", positive, row["category_id"]),
                ("neg", negative, negative_category_id),
            ):
                name = f"{stem}_{kind}{suffix}"
                window = Window(window_spec["col_off"], window_spec["row_off"], side, side)
                _write_crop(source, window, os.path.join(out_dir, name))

                records.append({
                    "image_name": name,
                    "img_path": out_dir.replace("\\", "/"),
                    "category_id": int(category),
                    "source_image": image_name,
                    "crop_kind": kind,
                    "side_px": int(side),
                    "annotations_on_tile": len(tile_rows),
                })

    result = pd.DataFrame(records)

    if verbose:
        pairs = len(result) // 2
        print(f"{pairs} matched pairs written to {out_dir}")
        print(f"  skipped, no room for a {MIN_SIDE_PX}px crop : {skipped_no_room}")
        print(f"  skipped, negative would clip the landform  : {skipped_overlap}")

    return result
