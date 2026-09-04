"""
Crop planning for the DeepLandforms landform tiles.

Why this exists
---------------
DeepLandforms cuts one tile per annotated landform, centred on that landform and
sized in proportion to it. Training on the tiles as they are hands a classifier
two shortcuts that have nothing to do with morphology:

* **Position.** The landform is always in the middle of the tile, so "is there a
  blob at the centre" separates a landform from anything else.
* **Tile size.** The tile side tracks the landform size (median tile/landform
  ratio 3.9), so the raster dimensions leak the class: the median annotated
  diameter is 155 m for Type-1, 512 m for Type-4 and 779 m for Type-2.

Two crop policies
-----------------
``plan_crop`` (**relative**, ``context`` x the annotation) removes the size cue
from the raster dimensions but reintroduces it as a *resampling* artefact once
the crop is resized to a fixed pixel side -- see ``plan_crop_fixed`` for the
measurement. ``plan_crop_fixed`` (**fixed footprint**) instead covers the same
ground extent every time and pads where the tile runs out. Report both: the gap
between them is how much of an optical score is morphology and how much is the
pipeline.

What the relative policy does
-----------------------------
One crop per annotation, from every tile, with:

* **side proportional to the landform** (``context`` x its bounding box), so the
  landform occupies a roughly constant fraction of the frame and the raster
  dimensions no longer encode the class;
* **a random offset**, constrained so the whole annotation stays inside the crop
  and the crop stays inside the tile, so the landform is not centred;
* **no rejection**. The side is clamped to the tile, and DeepLandforms tiles
  always contain their own landform (smallest measured tile/landform ratio
  1.38), so every one of the 1846 annotations yields a crop.

The bounding box is used rather than the segmentation polygon because in this
data set they agree: ``bbox`` is exactly the tight bound of the polygon
(total absolute disagreement 377 px over all 1846 annotations, i.e. rounding).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CropPlan",
    "plan_crop",
    "plan_crop_fixed",
    "padded_fraction",
    "read_crop",
    "landform_lonlat",
    "annotation_table",
    "export_crops",
]


class CropPlan(tuple):
    """A planned crop window: ``(col_off, row_off, side)`` in tile pixels."""

    __slots__ = ()

    def __new__(cls, col_off: int, row_off: int, side: int):
        return super().__new__(cls, (int(col_off), int(row_off), int(side)))

    col_off = property(lambda self: self[0])
    row_off = property(lambda self: self[1])
    side = property(lambda self: self[2])


def _offset_range(box_lo: float, box_len: float, side: int, extent: int) -> Tuple[int, int]:
    """
    Range of crop offsets along one axis that keep the annotation inside the crop
    and the crop inside the tile.

    Returns ``(lo, hi)`` with ``lo <= hi``. When the annotation is longer than
    the side -- possible only where the tile itself is smaller than the
    annotation -- the range collapses to the offset that centres the annotation,
    which loses the least of it.
    """
    lo = max(0.0, box_lo + box_len - side)
    hi = min(box_lo, extent - side)

    if lo > hi:
        centred = box_lo + box_len / 2.0 - side / 2.0
        centred = min(max(centred, 0.0), max(extent - side, 0))
        return int(round(centred)), int(round(centred))

    return int(np.ceil(lo)), int(np.floor(hi))


def plan_crop_fixed(
    tile_width: int,
    tile_height: int,
    bbox: Sequence[float],
    resolution_mpp: float,
    footprint_m: float = 960.0,
    rng: Optional[np.random.Generator] = None,
    jitter_fraction: float = 0.2,
) -> CropPlan:
    """
    Plan one square crop of a **fixed extent on the ground**, in tile pixels.

    Why this exists
    ---------------
    :func:`plan_crop` sizes the window as a multiple of the annotation and the
    dataset then resamples it to a fixed pixel side. That removes the landform's
    size from the raster dimensions, but it puts it straight back into the
    *pixels*: the resampling factor becomes a function of the landform's size,
    and landform size is strongly class-dependent. Measured over the 1846
    annotations, ``context=1.75`` gives a median effective ground sampling
    distance of 0.71 m/pixel for Type-1 against 3.45 m/pixel for Type-2, so
    Type-1 crops arrive upsampled and smooth while Type-2 crops arrive
    downsampled and sharp. That difference alone -- with no morphology at all --
    separates Type-1 from the rest at 80.9% accuracy under product-level
    grouping, against a 66.1% majority, and gradient statistics of the crop
    recover it with out-of-group :math:`R^2 = 0.59`.

    It is an artefact of the pipeline, not a property of Mars, and it would not
    survive a detector sliding over a native-resolution HiRISE product.

    What this does instead
    ----------------------
    Every crop covers the same ``footprint_m`` metres of ground. Paired with a
    fixed output ground sampling distance in the dataset, the resampling factor
    then depends only on the tile's own ``resolution_mpp`` -- which is the
    DeepLandforms replica axis and is very nearly balanced across classes
    (Type-1 is exactly 20% at each of 0.5/1/2/3/5 m/pixel, and
    ``resolution_mpp`` alone predicts the class at exactly the majority rate).

    The landform's *apparent* size now varies from crop to crop again. That is
    deliberate and it is not the same thing: apparent size at a fixed scale is a
    real physical cue that a real detector would also have, so it can be
    disclosed and ablated rather than accidentally manufactured.

    What it costs
    -------------
    A fixed footprint does not always fit the tile, and DeepLandforms tiles are
    themselves sized in proportion to their landform, so the padded fraction is
    class-dependent: at 960 m it averages 0.16 for Type-1 against 0.05 for
    Type-4 and 0.03 for Type-2. Padding alone predicts the class at 70.0%,
    against 80.9% for the effective GSD it replaces. It is not zero, so it is
    kept as a named control -- see ``model.baselines.geometry_features``.

    How the offset is drawn, and why not the way ``plan_crop`` draws it
    ------------------------------------------------------------------
    ``plan_crop`` picks uniformly from the offsets that keep the whole
    annotation inside the window. Carried over to a fixed footprint that rule
    becomes class-dependent, because a large landform nearly fills the window
    and so has almost nowhere to move: measured at 960 m, median available
    jitter would be 0.63 of the side for Type-1 against 0.18 for Type-2, with
    26% of Type-2 annotations unable to move at all. That is the positional
    shortcut handed straight back, class-dependently -- the same failure the
    ``context`` factor was tuned to avoid.

    So the fixed policy jitters the window *centre* by a fixed fraction of the
    footprint instead, uniformly and without reference to the annotation's size.
    Every sample gets exactly the same amount of positional augmentation, so the
    jitter cannot encode anything. The price is that a landform larger than
    ``(1 - 2 * jitter_fraction) * footprint_m`` can have an edge pushed out of
    frame -- which is what a detector scanning at a fixed scale would also see,
    and is therefore a property of the task rather than of the pipeline.

    :param tile_width: Tile width in pixels.
    :param tile_height: Tile height in pixels.
    :param bbox: ``[x, y, w, h]`` of the annotation, in tile pixels.
    :param resolution_mpp: Ground sampling distance of this tile, in metres.
    :param footprint_m: Side of the crop on the ground, in metres. 960 m is the
        default because it is where the padding leak is near its minimum while
        still containing the whole landform for 100% of Type-1, 95.7% of Type-4
        and 62.4% of Type-2 annotations. (960 m at 2.5 m/pixel also gives the
        same 384 px input as the relative policy, so the two cost the same to
        train.)
    :param rng: Source of randomness for the offset. ``None`` centres the window
        on the annotation -- previews only, never training.
    :param jitter_fraction: Half-width of the centre jitter, as a fraction of
        the footprint. 0.2 moves the landform centre anywhere in the middle 40%
        of the frame.
    :return: The planned window. ``col_off``/``row_off`` may be negative, and
        the window may extend past the tile; :func:`read_crop` reads boundless
        and pads with zeros.
    """
    box_x, box_y, box_w, box_h = (float(v) for v in bbox)

    side = max(int(round(footprint_m / float(resolution_mpp))), 1)

    # Window centred on the annotation, then displaced by the same amount for
    # everyone. No clamping to the tile: clamping is what would make the jitter
    # depend on the tile, and the tile depends on the landform.
    col_off = box_x + box_w / 2.0 - side / 2.0
    row_off = box_y + box_h / 2.0 - side / 2.0

    if rng is not None:
        jitter = int(round(abs(jitter_fraction) * side))
        if jitter > 0:
            col_off += int(rng.integers(-jitter, jitter + 1))
            row_off += int(rng.integers(-jitter, jitter + 1))

    return CropPlan(int(round(col_off)), int(round(row_off)), side)


def padded_fraction(
    plan: CropPlan, tile_width: int, tile_height: int
) -> float:
    """
    Share of a planned crop's area that falls outside the tile and is padded.

    Class-dependent under the fixed-footprint policy, so it is measured rather
    than assumed. Feeds ``model.baselines.geometry_features``.
    """
    inside_w = max(0, min(plan.col_off + plan.side, tile_width) - max(plan.col_off, 0))
    inside_h = max(0, min(plan.row_off + plan.side, tile_height) - max(plan.row_off, 0))
    return 1.0 - (inside_w * inside_h) / float(plan.side ** 2)


def plan_crop(
    tile_width: int,
    tile_height: int,
    bbox: Sequence[float],
    context: float = 1.75,
    rng: Optional[np.random.Generator] = None,
) -> CropPlan:
    """
    Plan one square crop around an annotation.

    :param tile_width: Tile width in pixels.
    :param tile_height: Tile height in pixels.
    :param bbox: ``[x, y, w, h]`` of the annotation, in tile pixels.
    :param context: Crop side as a multiple of the annotation's longest edge.
        1.75 puts the landform at ~57% of the frame, keeping its rim and a band
        of surrounding terrain in view.

        Do not raise this without re-checking the jitter it leaves. The crop is
        clamped to the tile, and the tile/landform ratio is itself
        class-correlated (median 3.9 overall, but Type-2 annotations are much
        larger relative to their tile), so a bigger context factor runs out of
        tile more often for some classes than others and hands the positional
        cue straight back. Measured share of annotations left with under 2% of
        the crop side to move in, by class (Type-1 / Type-4 / Type-2):

        =========  ==========================  =======================
        context    stuck (T1 / T4 / T2)        median jitter
        =========  ==========================  =======================
        1.75       1.0% / 2.4% / 7.4%          0.429 / 0.428 / 0.428
        2.0        2.7% / 13.9% / 19.7%        0.500 / 0.500 / 0.384
        2.5        17.4% / 24.4% / 36.0%       0.600 / 0.455 / 0.111
        =========  ==========================  =======================

        1.75 is the largest value at which the jitter is flat across classes.
    :param rng: Source of randomness for the offset. ``None`` centres the crop
        on the annotation -- use it only for previews, never for training data,
        because a centred landform is itself a shortcut.
    :return: The planned window.
    """
    box_x, box_y, box_w, box_h = (float(v) for v in bbox)
    longest = max(box_w, box_h)

    # As much context as asked for, but never past the tile, and never smaller
    # than the annotation itself.
    side = int(round(context * longest))
    side = max(side, int(np.ceil(longest)))
    side = min(side, tile_width, tile_height)
    side = max(side, 1)

    col_lo, col_hi = _offset_range(box_x, box_w, side, tile_width)
    row_lo, row_hi = _offset_range(box_y, box_h, side, tile_height)

    if rng is None:
        col_off = (col_lo + col_hi) // 2
        row_off = (row_lo + row_hi) // 2
    else:
        col_off = int(rng.integers(col_lo, col_hi + 1))
        row_off = int(rng.integers(row_lo, row_hi + 1))

    return CropPlan(col_off, row_off, side)


def read_crop(
    tile_path: str,
    plan: CropPlan,
    out_px: Optional[int] = None,
    resample: str = "bilinear",
) -> np.ndarray:
    """
    Read a planned crop out of a tile, optionally resampled to a fixed size.

    DeepLandforms tiles are single-band uint8, so nothing is rescaled here; the
    array is returned exactly as stored.

    :param tile_path: Path to the DeepLandforms GeoTIFF.
    :param plan: Window from :func:`plan_crop`.
    :param out_px: Side of the returned array. ``None`` keeps the native side.
    :param resample: ``"bilinear"`` or ``"nearest"``.
    :return: ``(out_px, out_px)`` uint8 array.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(tile_path) as src:
        patch = src.read(
            1,
            window=Window(plan.col_off, plan.row_off, plan.side, plan.side),
            boundless=True,
            fill_value=0,
        )

    if out_px is None or patch.shape[0] == out_px:
        return patch

    from PIL import Image

    mode = Image.BILINEAR if resample == "bilinear" else Image.NEAREST
    return np.asarray(Image.fromarray(patch).resize((out_px, out_px), mode))


def landform_lonlat(tile_path: str, bbox: Sequence[float]) -> Tuple[float, float]:
    """
    Centre of the *annotation* as (latitude, east longitude in 0..360).

    This is the landform's own position, not the tile's or the crop's. The
    thermal pipeline keys its windows on it: a THEMIS pixel is ~100 m and the
    median annotated landform is 280 m across, so a window centred on anything
    else misses the target. Tile centres happen to coincide here (DeepLandforms
    centres its tiles), but crop centres deliberately do not.
    """
    import rasterio

    from data.coordinates import _transformer_for

    box_x, box_y, box_w, box_h = (float(v) for v in bbox)

    with rasterio.open(tile_path) as src:
        x, y = src.xy(box_y + box_h / 2.0, box_x + box_w / 2.0)
        transformer = _transformer_for(src.crs.to_wkt())

    lon, lat = transformer.transform(x, y)
    return lat, lon % 360.0


def annotation_table(
    dataset_json: str,
    tile_dir: str,
    with_coordinates: bool = True,
    verbose: bool = True,
):
    """
    Flatten the DeepLandforms COCO-style ``dataset.json`` into one row per
    annotation, with everything the dataset and the split need.

    Columns
    -------
    image_name, img_path, category_id, bbox
        Tile identity, class and annotation box.
    landform_key
        The tile with its resolution token stripped, so the 0.5/1/2/3/5 m/pixel
        replicas of one landform share a key.
    product
        HiRISE product id. Two landforms in one product share season, solar
        incidence and calibration, so the split has to be able to hold them
        together.
    resolution_mpp, tile_width, tile_height
        Acquisition properties, kept so they can be checked as confounds rather
        than silently learned.
    lat, lon_east
        Centre of the annotation, for the thermal pipeline.

    :param dataset_json: Path to DeepLandforms ``dataset.json``.
    :param tile_dir: Directory holding the tiles.
    :param with_coordinates: Read each tile's georeferencing. Costs about a
        minute for the full set; without it there are no ``lat``/``lon_east``.
    """
    import json

    import pandas as pd

    with open(dataset_json) as handle:
        payload = json.load(handle)

    images = pd.DataFrame(payload["images"])
    annotations = pd.DataFrame(payload["annotations"])
    categories = {c["id"]: c["name"] for c in payload["categories"]}

    table = annotations.merge(
        images, left_on="image_id", right_on="id", suffixes=("", "_img")
    )
    table = table.rename(columns={"file_name": "image_name"})
    table["img_path"] = tile_dir.replace("\\", "/")
    table["category_name"] = table["category_id"].map(categories)

    table["landform_key"] = (
        table["image_name"]
        .str.replace(r"_resized_[0-9.]+", "", regex=True)
        .str.replace(".tiff", "", regex=False)
    )
    table["product"] = table["image_name"].str.extract(r"^((?:ESP|PSP)_\d+_\d+)")
    table["resolution_mpp"] = (
        table["image_name"].str.extract(r"_resized_([0-9.]+)_").astype(float)
    )
    table = table.rename(columns={"width": "tile_width", "height": "tile_height"})

    table["box_px"] = table["bbox"].apply(lambda b: max(float(b[2]), float(b[3])))
    table["diameter_m"] = table["box_px"] * table["resolution_mpp"]

    if with_coordinates:
        from tqdm.auto import tqdm

        coords = []
        for row in tqdm(
            table.itertuples(index=False),
            total=len(table),
            desc="Annotation coordinates",
        ):
            path = os.path.join(row.img_path, row.image_name)
            try:
                coords.append(landform_lonlat(path, row.bbox))
            except Exception:
                coords.append((np.nan, np.nan))
        table["lat"] = [c[0] for c in coords]
        table["lon_east"] = [c[1] for c in coords]

    keep = [
        "image_name", "img_path", "category_id", "category_name", "bbox",
        "landform_key", "product", "resolution_mpp",
        "tile_width", "tile_height", "box_px", "diameter_m",
    ]
    if with_coordinates:
        keep += ["lat", "lon_east"]
    table = table[keep].reset_index(drop=True)

    if verbose:
        print(f"{len(table)} annotations over {table.image_name.nunique()} tiles")
        print(f"  {table.landform_key.nunique()} landform keys, "
              f"{table['product'].nunique()} HiRISE products")
        print(table["category_name"].value_counts().to_string())

    return table


def export_crops(
    annotations,
    out_dir: str = "data/optical/landform_crops",
    context: float = 1.75,
    out_px: int = 384,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Materialise one crop per annotation as a georeferenced GeoTIFF.

    Training does **not** need this -- ``LandformDataset`` plans and cuts crops
    on the fly so the offset is redrawn every epoch, which is where the
    augmentation comes from. Use this to inspect the crops, to share a frozen
    snapshot, or to reproduce a figure.

    :return: The annotation rows describing the exported crops.
    """
    import pandas as pd
    import rasterio
    from affine import Affine
    from rasterio.windows import Window
    from tqdm.auto import tqdm

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []

    for row in tqdm(
        annotations.itertuples(index=False), total=len(annotations), desc="Crops"
    ):
        source = os.path.join(row.img_path, row.image_name)
        plan = plan_crop(row.tile_width, row.tile_height, row.bbox, context, rng)

        with rasterio.open(source) as src:
            window = Window(plan.col_off, plan.row_off, plan.side, plan.side)
            patch = src.read(1, window=window, boundless=True, fill_value=0)
            profile = src.profile.copy()
            transform = src.window_transform(window)

        if out_px and patch.shape[0] != out_px:
            from PIL import Image

            patch = np.asarray(
                Image.fromarray(patch).resize((out_px, out_px), Image.BILINEAR)
            )
            scale = plan.side / out_px
            transform = transform * Affine.scale(scale, scale)

        stem, suffix = os.path.splitext(row.image_name)
        name = f"{stem}_crop{suffix}"
        profile.update(
            width=patch.shape[1], height=patch.shape[0],
            transform=transform, dtype="uint8", count=1,
        )
        with rasterio.open(os.path.join(out_dir, name), "w", **profile) as sink:
            sink.write(patch, 1)

        records.append({
            **row._asdict(),
            "crop_name": name,
            "crop_dir": out_dir.replace("\\", "/"),
            "crop_side_px": plan.side,
            "crop_gsd_m": plan.side * row.resolution_mpp / (out_px or plan.side),
        })

    result = pd.DataFrame(records)
    if verbose:
        print(f"{len(result)} crops written to {out_dir}")
    return result
