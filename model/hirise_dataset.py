"""
Dataset over the DeepLandforms landform tiles.

Crops are planned and cut **on the fly** rather than read from a pre-cut
directory. That is deliberate: the crop offset is redrawn on every access, so a
landform is seen at a different position in the frame each epoch. Baking the
crops to disk would freeze one offset per sample and throw that away.

See ``data/optical/landform_crops.py`` for the crop policy and why the tiles
cannot be used as they are.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.optical.landform_crops import plan_crop, read_crop
from data.thermal.themis_windows import NODATA_KELVIN

__all__ = ["LandformDataset", "MARS_RADIUS_M"]

#: Mean Mars radius, for converting a merge radius to metres on the ground.
MARS_RADIUS_M = 3389500.0

#: Grouping relations available to :meth:`LandformDataset.group_keys`.
GROUP_LEVELS = ("landform", "product")


class LandformDataset(Dataset):
    """
    One sample per DeepLandforms annotation.

    :param annotations: Table from ``landform_crops.annotation_table`` -- one row
        per annotation, carrying ``image_name``, ``img_path``, ``category_id``,
        ``bbox``, ``tile_width``, ``tile_height``, ``landform_key``, ``product``
        and, where available, ``lat``/``lon_east``.
    :param root_dir: Prefix for ``img_path``.
    :param out_px: Side of the returned image, in pixels.
    :param context: Crop side as a multiple of the annotation's longest edge.
    :param augment: Redraw the crop offset on every access and apply random
        flips and quarter turns. Set for training, clear for validation.
    :param seed: With ``augment=False``, offsets are drawn from this seed and
        the index, so validation crops are random but identical across runs --
        the same distribution as training, without the epoch-to-epoch churn.
    :param harmonize_intensity: Per-image percentile stretch. See
        :meth:`stretch_intensity`.
    :param load_optical: Set ``False`` for thermal-only runs to skip decoding.
    :param thermal_sites: ``site_id`` per row, for the thermal branch. Labels
        are never read from the thermal store -- see :meth:`thermal_for`.
    :param thermal_dir: directory of per-site ``.npy`` windows. ``None`` keeps
        the ``torch.randn`` placeholder, so a modality ablation still runs.
    :param thermal_mean: Kelvin mean used to fill and standardise. ``None``
        derives it from the valid pixels of the store.
    :param thermal_std: as above, for the standard deviation.
    """

    def __init__(
        self,
        annotations: pd.DataFrame,
        root_dir: str = "",
        out_px: int = 384,
        context: float = 1.75,
        augment: bool = False,
        seed: int = 42,
        harmonize_intensity: bool = True,
        load_optical: bool = True,
        generate_synthetic_thermal: bool = True,
        sequence_length: int = 3,
        thermal_window: int = 32,
        thermal_sites: Optional[Sequence[str]] = None,
        thermal_dir: Optional[str] = None,
        thermal_mean: Optional[float] = None,
        thermal_std: Optional[float] = None,
    ):
        required = {"image_name", "img_path", "category_id", "bbox",
                    "tile_width", "tile_height"}
        missing = required - set(annotations.columns)
        if missing:
            raise ValueError(
                f"annotations is missing {sorted(missing)}; build it with "
                "data.optical.landform_crops.annotation_table"
            )

        self.root_dir = root_dir
        self.img_dirs = annotations["img_path"].tolist()
        self.img_names = annotations["image_name"].tolist()
        self.img_labels = annotations["category_id"].tolist()
        self.boxes = annotations["bbox"].tolist()
        self.tile_widths = annotations["tile_width"].tolist()
        self.tile_heights = annotations["tile_height"].tolist()

        self.landform_keys = (
            annotations["landform_key"].tolist()
            if "landform_key" in annotations.columns else list(self.img_names)
        )
        self.products = (
            annotations["product"].tolist()
            if "product" in annotations.columns else [None] * len(self.img_names)
        )

        has_coords = {"lat", "lon_east"}.issubset(annotations.columns)
        self.lat = annotations["lat"].tolist() if has_coords else None
        self.lon_east = annotations["lon_east"].tolist() if has_coords else None

        self.out_px = out_px
        self.context = context
        self.augment = augment
        self.seed = seed
        self.harmonize_intensity = harmonize_intensity
        self.load_optical = load_optical

        self.generate_synthetic_thermal = generate_synthetic_thermal
        self.sequence_length = sequence_length
        self.thermal_window = thermal_window
        self.thermal_sites = (
            list(thermal_sites) if thermal_sites is not None else None
        )
        self.thermal_dir = thermal_dir
        self.thermal_mean = thermal_mean
        self.thermal_std = thermal_std
        self._thermal_store = None

        if thermal_dir is not None and thermal_sites is None:
            raise ValueError(
                "thermal_dir needs thermal_sites: one site_id per annotation, "
                "from data.thermal.thermal_sites.landform_sites"
            )

    def __len__(self) -> int:
        return len(self.img_names)

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def group_keys(
        self, level: str = "product", merge_radius_m: float = 300.0
    ) -> List[str]:
        """
        Identity of the site behind each sample, for group-aware splitting.

        Three separate things put correlated samples in this dataset, and a
        split that ignores any of them reports an inflated score:

        1. **Resolution replicas.** DeepLandforms holds the same landform at
           0.5, 1, 2, 3 and 5 m/pixel. A random row split puts rescaled copies
           of one pit on both sides.
        2. **Repeat observations.** The same pit is often imaged on several
           orbits under different product ids, so the filenames give no hint
           they are the same feature. Caught by proximity on the ground.
        3. **Shared acquisition.** Two *different* landforms in one HiRISE
           product share season, solar incidence angle, calibration and local
           terrain. 159 products carry 1846 annotations, and one product alone
           carries 78.

        ``level="landform"`` applies relations 1 and 2 and yields ~312 groups.
        ``level="product"`` adds relation 3 and yields ~159. The second is the
        honest number to report; the first is worth reporting beside it, since
        the gap between them measures how much acquisition context the model is
        leaning on.

        :param level: ``"landform"`` or ``"product"``.
        :param merge_radius_m: Ground distance under which two samples are
            treated as the same site. Zero disables relation 2.
        """
        if level not in GROUP_LEVELS:
            raise ValueError(f"level must be one of {GROUP_LEVELS}, got {level!r}")

        parent: dict = {}

        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Relation 1: same landform, different resolution replica.
        for i, key in enumerate(self.landform_keys):
            union(i, f"name::{key}")

        # Relation 3: same HiRISE product.
        if level == "product":
            for i, product in enumerate(self.products):
                if product is not None and not pd.isna(product):
                    union(i, f"prod::{product}")

        # Relation 2: close enough on the ground to be the same site.
        if self.lat is not None and merge_radius_m > 0:
            try:
                from scipy.spatial import cKDTree
            except ImportError:
                cKDTree = None

            if cKDTree is not None:
                lat = np.radians(np.asarray(self.lat, dtype=float))
                lon = np.radians(np.asarray(self.lon_east, dtype=float))
                finite = np.isfinite(lat) & np.isfinite(lon)
                if finite.any():
                    # Local planar approximation, accurate well past the radius.
                    points = np.c_[
                        MARS_RADIUS_M * lon * np.cos(lat), MARS_RADIUS_M * lat
                    ]
                    index = np.flatnonzero(finite)
                    tree = cKDTree(points[index])
                    for a, b in tree.query_pairs(merge_radius_m):
                        union(int(index[a]), int(index[b]))

        roots: dict = {}
        return [
            roots.setdefault(find(i), f"site_{len(roots):05d}")
            for i in range(len(self.img_names))
        ]

    # ------------------------------------------------------------------
    # Thermal
    # ------------------------------------------------------------------

    def _load_thermal_store(self) -> None:
        """
        Read every ``.npy`` window once, and derive the normalisation from the
        *valid* pixels only.

        Deriving mean and standard deviation from valid pixels matters: 12.2% of
        pixels carry the archive's no-data value, and letting those into the
        statistics moves the mean by ~23 K and doubles the standard deviation.
        """
        import glob

        store, valid_values = {}, []
        for path in glob.glob(os.path.join(self.thermal_dir, "*.npy")):
            stack = np.load(path).astype(np.float32)
            if stack.ndim == 4:  # (T, 1, k, k) from an older writer
                stack = stack[:, 0]
            store[os.path.splitext(os.path.basename(path))[0]] = stack
            valid = stack[stack != NODATA_KELVIN]
            if valid.size:
                valid_values.append(valid)

        self._thermal_store = store

        if self.thermal_mean is None or self.thermal_std is None:
            pooled = np.concatenate(valid_values) if valid_values else np.array([0.0])
            self.thermal_mean = float(pooled.mean())
            self.thermal_std = float(pooled.std()) or 1.0

    def thermal_for(self, idx: int) -> torch.Tensor:
        """
        Thermal sequence for one sample, shaped ``(T, 2, k, k)``.

        The thermal store is keyed purely by ``site_id`` and holds **no labels**.
        The class always comes from the optical annotation table, so changing how
        the HiRISE side is cut, filtered or relabelled cannot silently change
        what the thermal branch is trained against.

        **Two channels, not one.** Channel 0 is standardised brightness
        temperature; channel 1 is a validity mask, 1 where a real measurement
        exists and 0 where there is none. Missing values in channel 0 are filled
        with the dataset mean, which standardises to exactly 0.

        Why not a sentinel like -1 in a single channel:

        * *It corrupts the statistics.* 12.2% of pixels are missing. Any
          out-of-range fill drags the batch mean 23 K and doubles the standard
          deviation, and ``ThermalEncoder`` runs ``BatchNorm2d``, so two sites
          with identical physics but different coverage would normalise
          differently. A mean fill is neutral by construction.
        * *Missingness is spatial, not just per-frame.* 13% of frames are
          partially valid -- holes inside the window. A 3x3 convolution over a
          sentinel hole beside real terrain emits a blend the network then has to
          learn to undo. The mask rides through the same convolutions and gives
          every layer a spatially aligned "how much of this was real".

        The mask does **not** remove the availability leak -- coverage still
        correlates with class, and the model can read the mask. It makes the leak
        a named input that can be ablated and measured, which is what
        ``model.baselines.availability_features`` exists to quantify.

        Frames are ordered by Mars local solar time and packed from the front;
        absent slots are all-fill with an all-zero mask.
        """
        k = self.thermal_window
        length = self.sequence_length

        if self.thermal_dir is None:
            # Placeholder: no real store attached. Kept so a modality ablation
            # can be run before the archive work lands, where "thermal" must
            # score at the majority class.
            return torch.randn(length, 2, k, k, dtype=torch.float32)

        if self._thermal_store is None:
            self._load_thermal_store()

        site = self.thermal_sites[idx] if self.thermal_sites is not None else None
        stack = self._thermal_store.get(site) if site is not None else None

        values = np.full((length, k, k), self.thermal_mean, dtype=np.float32)
        mask = np.zeros((length, k, k), dtype=np.float32)

        if stack is not None and len(stack):
            # A stored window is never smaller than the one asked for: extraction
            # uses the largest size under consideration and anything narrower is
            # a centred slice of it.
            side = stack.shape[-1]
            if side > k:
                lo = (side - k) // 2
                stack = stack[:, lo:lo + k, lo:lo + k]

            frames = stack[:length]
            valid = frames != NODATA_KELVIN
            values[:len(frames)] = np.where(valid, frames, self.thermal_mean)
            mask[:len(frames)] = valid.astype(np.float32)

        values = (values - self.thermal_mean) / self.thermal_std

        return torch.from_numpy(np.stack([values, mask], axis=1))

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def _rng(self, idx: int) -> np.random.Generator:
        # Training redraws every access, so a landform lands in a different part
        # of the frame each epoch. Validation is seeded from the index, so its
        # crops are drawn from the same distribution but never move.
        if self.augment:
            return np.random.default_rng()
        return np.random.default_rng([self.seed, idx])

    def __getitem__(self, idx: int):
        label = self.img_labels[idx]

        if not self.load_optical:
            # 1x1 placeholder: collates like a real image, costs nothing, and
            # the model discards it under modality="thermal".
            image = torch.zeros(1, 1, 1, dtype=torch.float32)
        else:
            rng = self._rng(idx)
            path = os.path.abspath(
                os.path.join(self.root_dir, self.img_dirs[idx], self.img_names[idx])
            )
            plan = plan_crop(
                self.tile_widths[idx],
                self.tile_heights[idx],
                self.boxes[idx],
                context=self.context,
                rng=rng,
            )
            patch = read_crop(path, plan, out_px=self.out_px)

            if self.harmonize_intensity:
                patch = self.stretch_intensity(patch)

            if self.augment:
                patch = self.random_dihedral(patch, rng)

            image = torch.from_numpy(
                np.ascontiguousarray(patch, dtype=np.float32) / 255.0
            ).unsqueeze(0)

        if self.generate_synthetic_thermal:
            return image, self.thermal_for(idx), label

        return image, label

    # ------------------------------------------------------------------
    # Pixel handling
    # ------------------------------------------------------------------

    @staticmethod
    def random_dihedral(
        image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Random flip and quarter turn (the 8 symmetries of the square).

        Note the caveat: this changes the apparent illumination direction, which
        on a real surface is fixed by the solar azimuth. What separates these
        classes -- shadow fraction, rim sharpness, whether a floor is visible --
        is orientation-invariant, so the transform is safe for *this* task, but
        it would not be for anything that reads slope direction.
        """
        if rng.integers(2):
            image = np.fliplr(image)
        if rng.integers(2):
            image = np.flipud(image)
        turns = int(rng.integers(4))
        if turns:
            image = np.rot90(image, turns)
        return np.ascontiguousarray(image)

    @staticmethod
    def stretch_intensity(
        image: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0
    ) -> np.ndarray:
        """
        Per-image percentile stretch onto 0-255.

        This removes the brightness offset between HiRISE products, which is
        driven mostly by solar incidence angle. That cuts both ways and is why
        it is a flag rather than a fixed step: incidence angle is a genuine
        acquisition confound here -- Nodjoumi et al. (2023) note that Type-1b,
        Type-2a and Type-4 are hardest to tell apart precisely when incidence is
        low -- but it is also physically tied to the shadow that makes a deep
        pit readable. Train both ways and report the difference.
        """
        arr = image.astype(np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return image

        low, high = np.percentile(finite, [low_pct, high_pct])
        if high <= low:
            return image

        # float32 throughout: np.percentile returns float64 scalars, and letting
        # them promote arr doubles the working set of every crop on the hot path.
        scale = np.float32(255.0 / (high - low))
        stretched = (arr - np.float32(low)) * scale
        np.clip(stretched, 0.0, 255.0, out=stretched)
        return stretched.astype(np.uint8)
