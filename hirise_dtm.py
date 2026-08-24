import os
import re
import tempfile
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import rasterio

plt.style.use('default')

# HiRISE special values and metadata saturation flags according to PDS standards
HIRISE_SPECIAL_VALUES = {0, 1, 2, 1022, 1023}


class HiriseDTM:
    """
    Handles memory-mapped ingestion, patch extraction, and masking for HiRISE
    _RED.JP2 orthoimages prepared for deep learning workflows.
    """

    def __init__(self, img_path: str | os.PathLike = None, img=None):
        self._temp_file_path = None  # Tracks temp file for cleanup

        if img_path:
            self.img_path = str(img_path)

            with rasterio.open(self.img_path) as src:
                shape = src.shape
                tf = tempfile.NamedTemporaryFile(delete=False, prefix='hirise_memmap_', suffix='.dat')
                self._temp_file_path = tf.name
                tf.close()

                # Read raw 16-bit uint container straight to memmap
                self.numpy_image = np.memmap(self._temp_file_path, dtype='uint16', mode='w+', shape=shape)
                src.read(1, out=self.numpy_image)

            # Flush modifications to disk
            self.numpy_image.flush()

            self.file_name = os.path.splitext(os.path.basename(self.img_path))[0]
            self.metadata = self._get_metadata()
            self.bounds = self._load_lbl_bounds()

        else:
            self.numpy_image = np.array(img, dtype='float32') if img is not None else None
            self.bounds = {}

    def _apply_initial_mask(self, nodata_val=None):
        """Replaces native nodata and HiRISE special flags with np.nan."""
        mask = np.isin(self.numpy_image, list(HIRISE_SPECIAL_VALUES))
        if nodata_val is not None:
            mask |= (self.numpy_image == nodata_val)

        self.numpy_image[mask] = np.nan

    def __del__(self):
        """Cleanup temporary memory-map file upon object deletion."""
        if hasattr(self, 'numpy_image') and isinstance(self.numpy_image, np.memmap):
            try:
                self.numpy_image._mmap.close()
            except Exception:
                pass
            del self.numpy_image

        if self._temp_file_path and os.path.exists(self._temp_file_path):
            try:
                os.remove(self._temp_file_path)
            except PermissionError:
                pass

    def _load_lbl_bounds(self) -> Dict[str, float]:
        """Parses the companion .LBL file for map parameters."""
        base_path = os.path.splitext(self.img_path)[0]
        lbl_path = base_path + '.LBL' if os.path.exists(base_path + '.LBL') else base_path + '.lbl'

        if not os.path.exists(lbl_path):
            return {"MAX_LAT": None, "MIN_LAT": None, "EAST_LON": None, "WEST_LON": None, "MAP_SCALE": None}

        bounds = {}
        patterns = {
            "MAX_LAT": r"MAXIMUM_LATITUDE\s*=\s*(-?\d+\.?\d*)",
            "MIN_LAT": r"MINIMUM_LATITUDE\s*=\s*(-?\d+\.?\d*)",
            "EAST_LON": r"EASTERNMOST_LONGITUDE\s*=\s*(-?\d+\.?\d*)",
            "WEST_LON": r"WESTERNMOST_LONGITUDE\s*=\s*(-?\d+\.?\d*)",
            "MAP_SCALE": r"MAP_SCALE\s*=\s*(-?\d+\.?\d*)"
        }

        with open(lbl_path, 'r') as f:
            content = f.read()
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                bounds[key] = float(match.group(1)) if match else None

        return bounds

    def get_portion_of_map(self, size: int, max_percentage_invalid: float = 0.0) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extracts a random size x size patch, strictly constraining the fraction of
        invalid/masked (NaN) pixels allowed inside the patch.

        max_percentage_invalid is between 0-1
        """
        img_height, img_width = self.numpy_image.shape[:2]
        max_attempts = 10000

        for _ in range(max_attempts):
            x = np.random.randint(0, img_width - size + 1)
            y = np.random.randint(0, img_height - size + 1)

            # Read only the tiny patch into RAM
            patch = self.numpy_image[y:y + size, x:x + size].astype('float32')

            # Mask invalid values on just this patch
            invalid_mask = np.isin(patch, list(HIRISE_SPECIAL_VALUES))
            patch[invalid_mask] = np.nan

            invalid_ratio = np.isnan(patch).mean()
            if invalid_ratio <= max_percentage_invalid:
                return patch, (y, x)

        raise RuntimeError("Could not find a valid patch meeting criteria.")

    def apply_nodata_patch(self, y: int, x: int, size: int) -> None:
        """
        Marks an extracted/seen tile region as invalid using np.nan
        to prevent future extraction overlapping this region.
        """
        self.numpy_image[y:y + size, x:x + size] = 0
        self.numpy_image.flush()

    def undo_nodata_patch(self, y: int, x: int, size: int) -> None:
        """Restores a masked region back to its original raw values from the source image."""
        with rasterio.open(self.img_path) as src:
            window = rasterio.windows.Window(x, y, size, size)
            original_data = src.read(1, window=window).astype('float32')
            nodata = src.nodata

            # Re-apply special flag and nodata masking
            mask = np.isin(original_data, list(HIRISE_SPECIAL_VALUES))
            if nodata is not None:
                mask |= (original_data == nodata)

            original_data[mask] = np.nan
            self.numpy_image[y:y + size, x:x + size] = original_data

        self.numpy_image.flush()

    def get_min_max_dn(self) -> Tuple[float, float]:
        """Returns the minimum and maximum valid Digital Number (DN) values."""
        return float(np.nanmin(self.numpy_image)), float(np.nanmax(self.numpy_image))

    def plot_dtm(self, dtm=None, figsize: Tuple[int, int] = (10, 10)) -> None:
        """Visualizes the image or patch with grayscale colormap suitable for orthoimages."""
        img_to_plot = dtm if dtm is not None else self.numpy_image
        plt.figure(figsize=figsize)
        plt.imshow(img_to_plot, cmap="gray")
        plt.colorbar(label="Digital Number (DN)")
        plt.title("HiRISE RED Orthoimage")
        plt.show()

    def _get_metadata(self) -> Dict:
        """Parses RDR or DTM metadata from the filename."""
        if not hasattr(self, 'file_name') or not self.file_name:
            return {}

        parts = self.file_name.split("_")

        if len(parts) == 6:
            aabcd, xxxxxx, xxxx, yyyyyy, yyyy, Vnn = parts
            return {
                "product_type": "DTM" if aabcd[:2] == "DT" else "unknown",
                "orbit_and_latitude_1": (xxxxxx, xxxx),
                "orbit_and_latitude_2": (yyyyyy, yyyy),
                "version_number": Vnn[1:]
            }
        elif len(parts) == 4:
            phase_or_type, orbit_number, lat_band, band_color = parts
            return {
                "product_type": "Ortho Image",
                "phase": phase_or_type,
                "orbit_number": orbit_number,
                "latitude_band": lat_band,
                "band_color": band_color
            }

        return {"product_type": "unknown", "raw_filename": self.file_name}

    def get_pixel_coordinate(self, row: int, col: int) -> Tuple[float, float]:
        """Returns projected coordinates (Easting, Northing) in meters for pixel (row, col)."""
        with rasterio.open(self.img_path) as src:
            x, y = src.transform * (col, row)
            return x, y

    def get_lat_lon(self, row: int, col: int) -> Tuple[float, float]:
        """Returns global Mars geographic coordinates (Latitude, Longitude) via LBL interpolation."""
        if not getattr(self, 'bounds', None) or any(v is None for v in self.bounds.values()):
            raise ValueError(
                f"LBL bounds are missing or incomplete for file {self.img_path}."
            )

        height, width = self.numpy_image.shape[:2]

        lat = self.bounds['MAX_LAT'] - (row / height) * (self.bounds['MAX_LAT'] - self.bounds['MIN_LAT'])
        lon = self.bounds['WEST_LON'] + (col / width) * (self.bounds['EAST_LON'] - self.bounds['WEST_LON'])

        return lat, lon

    def show_image_portion(
            self,
            portion: np.ndarray | Tuple[np.ndarray, Tuple[int, int]],
            figsize: Tuple[int, int] = (6, 6)
    ) -> None:
        """
        Plots an image patch extracted from get_portion_of_map().

        :param portion: NumPy array patch OR tuple (patch, (y, x)) as returned by get_portion_of_map().
        :param figsize: Tuple specifying the figure dimensions.
        """
        coords_title = ""

        # Check if the user passed the full tuple (patch, (y, x)) returned by get_portion_of_map()
        if isinstance(portion, tuple) and len(portion) == 2 and isinstance(portion[0], np.ndarray):
            patch, (y, x) = portion
            coords_title = f" | Top-Left (Y={y}, X={x})"
        elif isinstance(portion, np.ndarray):
            patch = portion
        else:
            raise ValueError(
                "Input must be either a NumPy array or the (array, (y, x)) tuple returned by get_portion_of_map()."
            )

        plt.figure(figsize=figsize)
        plt.imshow(patch, cmap="gray")
        plt.colorbar(label="Digital Number (DN)")
        plt.title(f"HiRISE Patch ({patch.shape[1]}x{patch.shape[0]}){coords_title}")
        plt.xlabel("Pixel Column (X)")
        plt.ylabel("Pixel Row (Y)")
        plt.show()