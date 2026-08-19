import os
import rasterio
import numpy as np
import tempfile
from typing import Tuple, Dict
from matplotlib import pyplot as plt
import rasterio.warp
from pyproj import CRS, Transformer
import math
import re

plt.style.use('default')


class HiriseDTM:
    """
    This class takes as input the path to a local HiRISE .IMG or .JP2 file, converts it into a NumPy array
    (backed by a memory-mapped file for optimization), and provides a set of utility functions.

    :param img_path: Path to a local HiRISE .IMG or .JP2 file (supports full paths like "data/DTMs/file.JP2").
    """

    def __init__(self, img_path: str | os.PathLike = None, img=None):
        self._temp_file_path = None  # Keep track of temp file to delete later

        if img_path:
            self.img_path = str(img_path)

            with rasterio.open(self.img_path) as src:
                # 1. Determine shape and verify nodata
                shape = src.shape
                nodata = src.nodata

                # 2. Create a temporary file on disk to hold the array data
                tf = tempfile.NamedTemporaryFile(delete=False, prefix='hirise_memmap_', suffix='.dat')
                self._temp_file_path = tf.name
                tf.close()

                # 3. Create a memory-mapped array (float32 saves 50% RAM vs float64)
                self.numpy_image = np.memmap(self._temp_file_path, dtype='float32', mode='w+', shape=shape)

                # 4. Read data directly from source into the memmap
                src.read(1, out=self.numpy_image)

            # 5. Handle nodata (Process infinite walls)
            if nodata is not None:
                mask = (self.numpy_image == nodata)
                if np.any(mask):
                    self.numpy_image[mask] = np.inf

            # 6. Flush changes to disk
            self.numpy_image.flush()

            # Extracted file name without extension regardless of format (.IMG, .JP2, .jp2, etc.)
            self.file_name = os.path.splitext(os.path.basename(self.img_path))[0]
            self.metadata = self._get_metadata()

            # Load companion .LBL boundaries into a class parameter
            self.bounds = self._load_lbl_bounds()

        else:
            data = np.array(img)
            self.numpy_image = data
            self.bounds = {}

    def __del__(self):
        """
        Cleanup: Ensure the temporary file is deleted when the object is destroyed.
        """
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
                pass  # Windows sometimes holds locks longer than expected

    def _load_lbl_bounds(self) -> Dict[str, float]:
        """
        Parses the associated companion .LBL file to retrieve exact geographic bounds
        and stores them as a class parameter. Handles case-insensitive file extensions.
        """
        base_path = os.path.splitext(self.img_path)[0]

        # Check for .LBL or .lbl extension
        lbl_path = base_path + '.LBL'
        if not os.path.exists(lbl_path):
            lbl_path = base_path + '.lbl'

        if not os.path.exists(lbl_path):
            return {"MAX_LAT": None, "MIN_LAT": None, "EAST_LON": None, "WEST_LON": None}

        bounds = {}
        keys_to_extract = {
            "MAX_LAT": r"MAXIMUM_LATITUDE\s*=\s*(-?\d+\.?\d*)",
            "MIN_LAT": r"MINIMUM_LATITUDE\s*=\s*(-?\d+\.?\d*)",
            "EAST_LON": r"EASTERNMOST_LONGITUDE\s*=\s*(-?\d+\.?\d*)",
            "WEST_LON": r"WESTERNMOST_LONGITUDE\s*=\s*(-?\d+\.?\d*)"
        }

        with open(lbl_path, 'r') as f:
            content = f.read()
            for key, pattern in keys_to_extract.items():
                match = re.search(pattern, content)
                bounds[key] = float(match.group(1)) if match else None

        return bounds

    def get_portion_of_map(self, size, max_percentage_inf=0):
        """
        Extracts a size x size portion of the image, strictly avoiding
        infinities, NaNs, and already-patched (seen) areas.
        """
        img_height, img_width = self.numpy_image.shape[:2]
        max_attempts = 10000
        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            # Pick random top-left corner
            x = np.random.randint(0, img_width - size + 1)
            y = np.random.randint(0, img_height - size + 1)

            # Extract portion
            image_subset = np.array(self.numpy_image[y:y + size, x:x + size])

            # Count both zeros, infinities (user patches / native nodata) and NaNs
            num_invalid = np.sum(
                (image_subset == 0) | np.isinf(image_subset) | np.isnan(image_subset)
            )

            if num_invalid <= max_percentage_inf * (size * size):
                return image_subset, (y, x)

        raise RuntimeError(
            "Could not find a valid, unseen portion of the map. "
            "The map may be nearly fully covered or saturated with no-data values."
        )

    def apply_nodata_patch(self, y, x, size):
        """
        Given a coordinate (upper left corner) and a patch size,
        fills the area with zeros to mask it out as invalid/seen.
        """
        zero_fill = np.full(shape=(size, size), fill_value=0.0, dtype='float32')
        self.numpy_image[y:y + size, x:x + size] = zero_fill
        self.numpy_image.flush()

    def get_lowest_highest_altitude(self):
        return np.nanmin(self.numpy_image), np.nanmax(self.numpy_image)

    def plot_dtm(self, dtm=None, figsize: Tuple = (12, 12)) -> None:
        """
        :param dtm: dtm to plot (optional), if set to None, the whole map will be plotted.
        :param figsize: plot figure map_size.

        Shows the DTM numpy_image in a matplotlib figure.
        """
        img_to_plot = dtm if dtm is not None else self.numpy_image
        plt.figure(figsize=figsize)
        plt.imshow(img_to_plot, cmap="terrain")
        plt.colorbar(label="Elevation (m)")
        plt.title("HiRISE DTM")
        plt.show()

    def _get_metadata(self) -> Dict:
        """
        Returns the metadata of a HiRISE .IMG or .JP2 file based on standard naming conventions.
        """
        unk = "unknown"
        if not hasattr(self, 'file_name') or not self.file_name:
            return {}

        parts = self.file_name.split("_")

        # Case 1: Standard HiRISE DTM naming convention (6 parts)
        if len(parts) == 6:
            aabcd, xxxxxx, xxxx, yyyyyy, yyyy, Vnn = parts

            product_type = "DTM" if aabcd[:2] == "DT" else unk
            file_type = "Areoid Elevations" if aabcd[2] == "E" else unk
            projection = "Equirectangular" if aabcd[3] == "E" else "Polar Stereographic" if aabcd[3] == "P" else unk

            grid_spacing = (
                0.25 if aabcd[4] == "A" else
                0.5 if aabcd[4] == "B" else
                1.0 if aabcd[4] == "C" else
                2.0 if aabcd[4] == "D" else unk
            )

            producing_institution = {
                "U": "USGS",
                "A": "University of Arizona",
                "C": "CalTech",
                "N": "NASA Ames",
                "J": "JPL",
                "O": "Ohio State",
                "P": "Planetary Science Institute"
            }.get(Vnn[0], unk)

            return {
                "product_type": product_type,
                "file_type": file_type,
                "projection": projection,
                "grid_spacing": grid_spacing,
                "orbit_and_latitude_1": (xxxxxx, xxxx),
                "orbit_and_latitude_2": (yyyyyy, yyyy),
                "producing_institution": producing_institution,
                "version_number": Vnn[1:]
            }

        # Case 2: Standard HiRISE RDR / Ortho Image JP2 naming convention (4 parts)
        elif len(parts) == 4:
            phase_or_type, orbit_number, lat_band, band_color = parts

            return {
                "product_type": "Ortho Image",
                "phase": phase_or_type,
                "orbit_number": orbit_number,
                "latitude_band": lat_band,
                "band_color": band_color
            }

        else:
            return {
                "product_type": unk,
                "raw_filename": self.file_name
            }

    def get_pixel_coordinate(self, row: int, col: int) -> Tuple[float, float]:
        """
        Returns the projected coordinates (Easting, Northing) in meters for a pixel (row, col).
        """
        with rasterio.open(self.img_path) as src:
            x, y = src.transform * (col, row)
            return x, y

    def get_lat_lon(self, row: int, col: int) -> Tuple[float, float]:
        """
        Returns global Mars geographic coordinates (Latitude, Longitude)
        for a pixel (row, col) using the saved .LBL bounds.
        """
        if not getattr(self, 'bounds', None) or any(v is None for v in self.bounds.values()):
            raise ValueError(
                f"LBL bounds are missing or incomplete. Ensure the companion .LBL file "
                f"is saved in the same directory as {self.img_path}."
            )

        height, width = self.numpy_image.shape[:2]

        # Linear interpolation across the verified PDS image extent
        lat = self.bounds['MAX_LAT'] - (row / height) * (self.bounds['MAX_LAT'] - self.bounds['MIN_LAT'])
        lon = self.bounds['WEST_LON'] + (col / width) * (self.bounds['EAST_LON'] - self.bounds['WEST_LON'])

        return lat, lon