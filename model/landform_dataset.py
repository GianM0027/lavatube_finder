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

from data.optical.landform_crops import (padded_fraction, plan_crop,
                                         plan_crop_fixed, read_crop)
from data.thermal.themis_windows import NODATA_KELVIN

__all__ = ["LandformDataset", "MARS_RADIUS_M", "THERMAL_TIME_DIM"]

#: Mean Mars radius, for converting a merge radius to metres on the ground.
MARS_RADIUS_M = 3389500.0

#: Kelvin per unit after per-window normalisation. Fixed rather than derived
#: per window so that the *amplitude* of an anomaly survives: measured spread of
#: median-subtracted THEMIS pixels is 3.5 K, so 3 K keeps values near unit scale
#: while leaving the numbers physically interpretable.
PER_WINDOW_SCALE_K = 3.0

#: Grouping relations available to :meth:`LandformDataset.group_keys`.
GROUP_LEVELS = ("landform", "product")

#: Crop policies available to :class:`LandformDataset`. See
#: ``data.optical.landform_crops.plan_crop_fixed`` for why there are two.
CROP_POLICIES = ("relative", "fixed_gsd")

#: Width of the per-frame thermal time vector produced by
#: :meth:`LandformDataset.thermal_time_for`: sin/cos of local solar time,
#: sin/cos of solar longitude, and a flag saying whether either was known.
THERMAL_TIME_DIM = 5


class LandformDataset(Dataset):
    """
    One sample per DeepLandforms annotation.

    :param annotations: Table from ``landform_crops.annotation_table`` -- one row
        per annotation, carrying ``image_name``, ``img_path``, ``category_id``,
        ``bbox``, ``tile_width``, ``tile_height``, ``landform_key``, ``product``
        and, where available, ``lat``/``lon_east``.
    :param root_dir: Prefix for ``img_path``.
    :param out_px: Side of the returned image, in pixels. Ignored under
        ``crop_policy="fixed_gsd"``, where it is ``footprint_m / out_gsd_m``.
    :param crop_policy: ``"relative"`` sizes the crop as ``context`` x the
        annotation and resamples to ``out_px``; ``"fixed_gsd"`` covers
        ``footprint_m`` metres of ground and resamples to ``out_gsd_m`` metres
        per pixel. **The two answer different questions and both should be
        reported.** Under ``"relative"`` the resampling factor is a function of
        the landform's size, and landform size is strongly class-dependent, so
        the crop's effective ground sampling distance carries the class on its
        own: 80.9% accuracy on Type-1-versus-rest under product-level grouping,
        against a 66.1% majority, with no morphology involved. ``"fixed_gsd"``
        removes that at the cost of a class-dependent *padded* fraction, which
        carries 70.0% -- smaller, and named, so it can be ablated. See
        ``data.optical.landform_crops.plan_crop_fixed``.
    :param footprint_m: Ground extent of a ``"fixed_gsd"`` crop, in metres.
    :param out_gsd_m: Ground sampling distance of a ``"fixed_gsd"`` crop, in
        metres per pixel. The default pair, 960 m at 2.5 m/pixel, gives the same
        384 px input as the relative policy, so the two are compared at equal
        compute.
    :param context: Crop side as a multiple of the annotation's longest edge.
        ``"relative"`` policy only.
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
    :param thermal_normalisation: ``"per_window"`` (default) or ``"global"``.
        See :meth:`thermal_for` -- the choice decides whether the temperature
        channel carries a cave anomaly or a map reference.
    :param thermal_times: ``{site_id: [(lmst_hours, solar_longitude_deg), ...]}``
        from ``data.thermal.collect.frame_times``, in the same frame order as
        the stored windows. Without it the time vector is all zeros and the
        thermal branch is blind to the clock -- see :meth:`thermal_time_for`.
    :param cache_crops: keep each decoded crop in memory after its first read.
        **Only valid with** ``augment=False``, where ``__getitem__`` is a pure
        function of the index -- the offset comes from ``seed`` and ``idx`` and
        there is no dihedral transform -- so a cached crop is byte-identical to
        a freshly decoded one.

        Worth having because validation re-reads the same crops every epoch and
        decoding is the bottleneck: a fixed-footprint window is up to 1920 px on
        a side before it is resampled, at about 62 ms each. Cached as ``uint8``
        at ``out_px``, the whole 1846-row table costs roughly 270 MB, which is a
        fraction of what one dataloader worker process costs on Windows.
    """

    def __init__(
        self,
        annotations: pd.DataFrame,
        root_dir: str = "",
        out_px: int = 384,
        crop_policy: str = "relative",
        footprint_m: float = 960.0,
        out_gsd_m: float = 2.5,
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
        thermal_normalisation: str = "per_window",
        thermal_times: Optional[dict] = None,
        cache_crops: bool = False,
    ):
        if crop_policy not in CROP_POLICIES:
            raise ValueError(
                f"crop_policy must be one of {CROP_POLICIES}, got {crop_policy!r}"
            )

        required = {"image_name", "img_path", "category_id", "bbox",
                    "tile_width", "tile_height"}
        if crop_policy == "fixed_gsd":
            # The fixed-footprint plan is in metres, so it cannot be made
            # without knowing what a tile pixel is worth on the ground.
            required = required | {"resolution_mpp"}
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
        self.resolutions = (
            annotations["resolution_mpp"].astype(float).tolist()
            if "resolution_mpp" in annotations.columns
            else [None] * len(self.img_names)
        )

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

        self.crop_policy = crop_policy
        self.footprint_m = float(footprint_m)
        self.out_gsd_m = float(out_gsd_m)
        # Under the fixed policy the output side is not free: it is the
        # footprint divided by the target ground sampling distance, and letting
        # a caller set it independently would silently reintroduce a second,
        # invisible resampling step.
        self.out_px = (
            max(int(round(self.footprint_m / self.out_gsd_m)), 1)
            if crop_policy == "fixed_gsd" else out_px
        )
        self.context = context
        self.augment = augment

        if cache_crops and augment:
            raise ValueError(
                "cache_crops needs augment=False: with augmentation the crop "
                "offset and dihedral transform are redrawn on every access, so "
                "caching one draw would silently disable the augmentation"
            )
        self.cache_crops = cache_crops
        self._crop_cache = {} if cache_crops else None
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
        self.thermal_times = thermal_times
        self._thermal_store = None

        if thermal_normalisation not in ("per_window", "global"):
            raise ValueError(
                "thermal_normalisation must be 'per_window' or 'global', "
                f"got {thermal_normalisation!r}"
            )
        self.thermal_normalisation = thermal_normalisation

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

        ``level="landform"`` applies relations 1 and 2 and yields **318**
        groups. ``level="product"`` adds relation 3 and yields **127** -- fewer
        than the 159 distinct product ids, because the relations are combined by
        union-find and a landform observed on two orbits pulls both products
        into one component. The second is the honest number to report; the first
        is worth reporting beside it, since the gap between them measures how
        much acquisition context the model is leaning on.

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

        **Two channels, not one.** Channel 0 is brightness temperature; channel
        1 is a validity mask, 1 where a real measurement exists and 0 where
        there is none. Missing values are filled so they normalise to 0.

        **Normalisation decides what channel 0 means.** Under ``"global"`` it is
        standardised against the whole data set, so its dominant variance is
        *where and when* the frame was taken -- latitude, season, time of day.
        Measured: a model given only that scored AUC 0.671, statistically
        indistinguishable from one given only latitude and longitude (0.660). It
        was reading a map, not a cave.

        Under ``"per_window"`` (the default) each frame has its own median
        subtracted and is divided by a fixed 3 K. Everything acting on the whole
        neighbourhood equally cancels, and what survives is the local anomaly --
        which over 312 independent landforms separates skylights from other pits
        by 0.57 sigma with the sign Cushing et al. (2007) predict.

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

        if self.thermal_normalisation == "per_window":
            # Each frame is expressed relative to its own median, then scaled by
            # a FIXED number of Kelvin. Dividing by each window's own spread
            # would erase the amplitude, which is the quantity of interest.
            for t in range(length):
                real = mask[t] > 0
                if real.any():
                    values[t] = values[t] - np.median(values[t][real])
                else:
                    values[t] = 0.0
            values = values / PER_WINDOW_SCALE_K
        else:
            values = (values - self.thermal_mean) / self.thermal_std

        # Whatever the mode, a pixel with no measurement must arrive as exactly
        # zero: it was filled with a stand-in, and any residue of that stand-in
        # is a number the network can read as if it were a measurement.
        values[mask == 0] = 0.0

        return torch.from_numpy(np.stack([values, mask], axis=1))

    def thermal_time_for(self, idx: int) -> torch.Tensor:
        """
        When each thermal frame was taken, shaped ``(T, THERMAL_TIME_DIM)``.

        Five numbers per frame: ``sin`` and ``cos`` of Mars local solar time on
        the 24 h circle, ``sin`` and ``cos`` of solar longitude on the 360 deg
        circle, and a flag that is 1 where a real time was known and 0 for an
        absent frame.

        **Why this is a separate return value and not two more image channels.**
        Local solar time is a property of the frame, not of any pixel in it.
        Broadcasting it across a 32x32 window would push four constant planes
        through three convolutions and a batch norm, and then the centre-minus-
        annulus difference that ``ThermalEncoder`` computes would cancel most of
        what they carry. Concatenating them to the per-frame embedding, just
        before the temporal convolution, puts the clock exactly where the
        temporal model needs it and nowhere else.

        **Why the model needs a clock at all.** Cushing et al. (2007) identify a
        cave-connected pit by its *diurnal amplitude*: a temperature difference
        divided by a time difference. Without the second term there is no
        amplitude to compute. Worse, frames are packed from the front in local
        time order, so slot 0 is 03h at one site and 07h at the next -- the
        temporal convolution was comparing frames that are not comparable.

        And the regimes it is averaging over have *opposite signs*. Measured on
        the retrieved windows, the centre-minus-annulus contrast for Type-1
        against the other classes runs +1.27 K / +0.75 K / -0.27 K at 02-06 h
        (Cushing's warm insulated skylight) but -0.44 K / -3.53 K / -1.38 K at
        14-18 h, where a large Type-2 bowl is cold because it is *in shadow*.
        Those are two different physical mechanisms, and a model with no clock
        has to average them into one.

        Angles are encoded as sin/cos pairs rather than raw hours so that 23:30
        and 00:30 sit next to each other instead of 23 hours apart.

        With no ``thermal_times`` attached the whole tensor is zeros, which is
        also the ablation: pass ``thermal_times=None`` to reproduce the
        time-blind model.
        """
        length = self.sequence_length
        features = np.zeros((length, THERMAL_TIME_DIM), dtype=np.float32)

        if self.thermal_times is None or self.thermal_sites is None:
            return torch.from_numpy(features)

        rows = self.thermal_times.get(self.thermal_sites[idx]) or []

        for t, entry in enumerate(rows[:length]):
            hour, solar_longitude = entry
            known = False

            if hour is not None and np.isfinite(hour):
                angle = 2.0 * np.pi * (float(hour) % 24.0) / 24.0
                features[t, 0] = np.sin(angle)
                features[t, 1] = np.cos(angle)
                known = True

            if solar_longitude is not None and np.isfinite(solar_longitude):
                angle = np.radians(float(solar_longitude) % 360.0)
                features[t, 2] = np.sin(angle)
                features[t, 3] = np.cos(angle)
                known = True

            features[t, 4] = 1.0 if known else 0.0

        return torch.from_numpy(features)

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def plan_for(self, idx: int, rng: Optional[np.random.Generator] = None):
        """
        The crop window for one sample, under whichever policy is configured.

        Exposed because the geometry it produces -- the padded fraction, the
        effective ground sampling distance -- is exactly what the control
        baselines in ``model.baselines`` have to measure, and re-deriving it
        there would let the two drift apart.
        """
        if self.crop_policy == "fixed_gsd":
            return plan_crop_fixed(
                self.tile_widths[idx],
                self.tile_heights[idx],
                self.boxes[idx],
                resolution_mpp=self.resolutions[idx],
                footprint_m=self.footprint_m,
                rng=rng,
            )

        return plan_crop(
            self.tile_widths[idx],
            self.tile_heights[idx],
            self.boxes[idx],
            context=self.context,
            rng=rng,
        )

    def crop_geometry(self, idx: int) -> dict:
        """
        Geometry of one sample's crop: padded fraction and effective GSD.

        Both are potential shortcuts and neither is morphology, so they are
        reported rather than assumed. Under ``"relative"`` the effective GSD is
        the leak (80.9% on binary, product-grouped); under ``"fixed_gsd"`` it is
        constant by construction and the padded fraction takes over, at 70.0%.
        """
        plan = self.plan_for(idx, rng=None)
        resolution = self.resolutions[idx]
        return {
            "padded_fraction": padded_fraction(
                plan, self.tile_widths[idx], self.tile_heights[idx]
            ),
            "effective_gsd_m": (
                plan.side * resolution / self.out_px
                if resolution is not None else float("nan")
            ),
            "crop_side_px": float(plan.side),
        }

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
        elif self._crop_cache is not None and idx in self._crop_cache:
            image = torch.from_numpy(
                self._crop_cache[idx].astype(np.float32) / 255.0
            ).unsqueeze(0)
        else:
            rng = self._rng(idx)
            path = os.path.abspath(
                os.path.join(self.root_dir, self.img_dirs[idx], self.img_names[idx])
            )
            plan = self.plan_for(idx, rng)
            patch = read_crop(path, plan, out_px=self.out_px)

            if self.harmonize_intensity:
                patch = self.stretch_intensity(patch)

            if self.augment:
                patch = self.random_dihedral(patch, rng)

            if self._crop_cache is not None:
                # uint8 rather than the float tensor: a quarter of the memory,
                # and the division below is cheap.
                self._crop_cache[idx] = np.ascontiguousarray(patch, dtype=np.uint8)

            image = torch.from_numpy(
                np.ascontiguousarray(patch, dtype=np.float32) / 255.0
            ).unsqueeze(0)

        if self.generate_synthetic_thermal:
            # Four elements, not three: the thermal *sequence* and the times it
            # was taken at travel together, because neither means much without
            # the other. See :meth:`thermal_time_for`.
            return (
                image,
                self.thermal_for(idx),
                self.thermal_time_for(idx),
                label,
            )

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
