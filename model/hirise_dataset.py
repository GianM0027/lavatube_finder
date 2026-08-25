import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# DeepLandforms replicates each labelled landform across several spatial
# resolutions ("..._resized_2_lbl_3.tiff"). Stripping the resolution token
# recovers the identity of the underlying landform.
_RESOLUTION_TOKEN = re.compile(r"_resized_[0-9.]+")

# Matched crops (matched_crops.py) end in _pos / _neg. A pair comes out of one
# tile, so both halves must land on the same side of a split -- otherwise the
# tile's illumination and calibration leak from train into validation.
_CROP_KIND_TOKEN = re.compile(r"_(?:pos|neg)$")

# Mean Mars radius, for converting the merge radius to degrees of arc.
MARS_RADIUS_M = 3389500.0


class Hirise_Dataset(Dataset):
    def __init__(
            self,
            annotations_dataframe: pd.DataFrame,
            root_dir: str = "",  # Add root_dir (e.g. data_path or absolute project directory)
            transform=None,
            target_transform=None,
            generate_synthetic_thermal: bool = True,
            sequence_length: int = 3,
            thermal_window: int = 32,
            harmonize_intensity: bool = True,
            load_optical: bool = True
    ):
        self.root_dir = root_dir
        self.img_dirs = annotations_dataframe["img_path"].tolist()
        self.img_names = annotations_dataframe["image_name"].tolist()
        self.img_labels = annotations_dataframe["category_id"].tolist()

        # Optional; present once data_overview.ipynb has attached coordinates.
        # Used by group_keys to catch repeat observations of the same site.
        has_coords = {"lat", "lon_east"}.issubset(annotations_dataframe.columns)
        self.lat = annotations_dataframe["lat"].tolist() if has_coords else None
        self.lon_east = annotations_dataframe["lon_east"].tolist() if has_coords else None

        self.transform = transform
        self.target_transform = target_transform
        self.generate_synthetic_thermal = generate_synthetic_thermal
        self.sequence_length = sequence_length
        # Side length of the THEMIS window in native ~100 m/pixel samples.
        self.thermal_window = thermal_window
        # Put both source pipelines on a common intensity footing (see
        # stretch_intensity). Disable only to reproduce the previous behaviour.
        self.harmonize_intensity = harmonize_intensity
        # Set False for thermal-only training: decoding the multi-megapixel
        # HiRISE crops dominates loading time and none of it would be used.
        self.load_optical = load_optical

    def __len__(self) -> int:
        return len(self.img_names)

    @staticmethod
    def landform_key(image_name: str) -> str:
        """
        Reduce a filename to the tile it came from.

        Strips the resolution token DeepLandforms uses for its replicas and the
        _pos/_neg suffix of a matched crop pair, so every image derived from one
        tile shares a key. Names without those tokens are returned unchanged.
        """
        stem, _ = os.path.splitext(image_name)
        stem = _RESOLUTION_TOKEN.sub("", stem)
        return _CROP_KIND_TOKEN.sub("", stem)

    def group_keys(self, merge_radius_m: float = 300.0) -> list:
        """
        Identity of the site behind each sample, for group-aware splitting.

        Two separate things put copies of the same landform in the dataset, and
        both have to be collapsed or the validation score is inflated:

        1. **Resolution replicas.** DeepLandforms holds ~4.7 images of every
           labelled landform, identical apart from spatial resolution. Splitting
           rows at random put rescaled copies of the same pit on both sides
           (measured: 73.8% of validation images).
        2. **Repeat observations.** The same site was often imaged on several
           orbits and appears under different product IDs, so the filenames give
           no hint they are the same pit (e.g. ESP_016411_1605_lbl_0 and
           ESP_024481_1605_lbl_0 are one pit at -19.4646, 237.5546). 98 of 393
           filename groups collide this way, leaving 23.6% residual leakage.

        Neither key alone is enough: filenames miss repeat observations, and
        pure spatial clustering *splits* resolution replicas, whose tile centres
        drift slightly between resolutions. So samples are grouped by the union
        of the two relations -- same landform filename OR within
        ``merge_radius_m`` on the ground.

        Falls back to filenames alone when coordinates are unavailable.
        """
        keys = [self.landform_key(name) for name in self.img_names]

        if self.lat is None or merge_radius_m <= 0:
            return keys

        parent = {}

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

        # Relation 1: identical landform filename.
        for i, key in enumerate(keys):
            union(i, f"name::{key}")

        # Relation 2: close enough on the ground to be the same site.
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return [find(i) for i in range(len(keys))]

        radius = MARS_RADIUS_M
        lat = np.radians(np.asarray(self.lat, dtype=float))
        lon = np.radians(np.asarray(self.lon_east, dtype=float))
        # Local planar approximation; accurate well past the merge radius.
        points = np.c_[radius * lon * np.cos(lat), radius * lat]

        for i, j in cKDTree(points).query_pairs(merge_radius_m):
            union(i, j)

        roots = {}
        return [roots.setdefault(find(i), f"site_{len(roots):05d}") for i in range(len(keys))]

    def __getitem__(self, idx: int):
        label = self.img_labels[idx]

        if self.load_optical:
            # Resolve full path using absolute root_dir
            img_path = os.path.abspath(
                os.path.join(self.root_dir, self.img_dirs[idx], self.img_names[idx])
            )

            raw_img = Image.open(img_path)
            uint8_np = self.scale_hirise_to_uint8(np.array(raw_img))
            if self.harmonize_intensity:
                uint8_np = self.stretch_intensity(uint8_np)
            image = Image.fromarray(uint8_np)

            if self.transform:
                image = self.transform(image)
        else:
            # 1x1 placeholder: collates and batches like a real image, costs
            # nothing, and the model discards it under modality="thermal".
            image = torch.zeros(1, 1, 1, dtype=torch.float32)

        if self.target_transform:
            label = self.target_transform(label)

        if self.generate_synthetic_thermal:
            # Placeholder until real THEMIS windows are wired in. Shaped like the
            # real thing: a small native-resolution patch, NOT the HiRISE size.
            k = self.thermal_window
            thermal = torch.randn(self.sequence_length, 1, k, k, dtype=torch.float32)
            return image, thermal, label

        return image, label

    @staticmethod
    def stretch_intensity(
        image: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0
    ) -> np.ndarray:
        """
        Per-image percentile stretch, applied to every sample regardless of source.

        The two datasets reach `scale_hirise_to_uint8` in different states:
        DeepLandforms tiles are already uint8 and pass through untouched, while
        plain-terrain crops are raw float32 DN and get rescaled through the
        3-1021 clip. That left them on visibly different footings (median
        intensity ~100 for plain terrain against ~123-166 for the pit classes),
        which a network can exploit to separate the classes without ever
        looking at the landform. Stretching each image onto its own 2nd-98th
        percentile range removes that offset while preserving local contrast,
        which is what actually carries the morphology.
        """
        arr = image.astype(np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return image

        low, high = np.percentile(finite, [low_pct, high_pct])
        if high <= low:
            return image

        stretched = np.clip((arr - low) / (high - low), 0.0, 1.0) * 255.0
        return stretched.astype(np.uint8)

    @staticmethod
    def scale_hirise_to_uint8(image: np.ndarray, fill_value: int = 0) -> np.ndarray:
        if image.dtype == np.uint8:
            return image

        max_val = np.nanmax(image)
        if max_val <= 255.0:
            invalid_mask = np.isnan(image)
            uint8_img = np.nan_to_num(image, nan=fill_value).clip(0, 255).astype(np.uint8)
            uint8_img[invalid_mask] = fill_value
            return uint8_img

        invalid_mask = (
            np.isnan(image) |
            np.isin(image, [0, 1, 2, 1022, 1023]) |
            (image < 3) |
            (image > 1021)
        )
        clipped = np.clip(image, 3.0, 1021.0)
        scaled = ((clipped - 3.0) / (1021.0 - 3.0)) * 255.0
        uint8_img = np.round(scaled).astype(np.uint8)
        uint8_img[invalid_mask] = fill_value
        return uint8_img