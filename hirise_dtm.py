import os
import rasterio
import numpy as np
import tempfile
from typing import Tuple, Dict
from matplotlib import pyplot as plt
from typing import Dict

plt.style.use('default')

class HiriseDTM:
    """
    This class takes as input the path to a local HiRISE .IMG or .JP2 file, converts it into a NumPy array
    (backed by a memory-mapped file for optimization), and provides a set of utility functions.

    :param img_path: Path to a local HiRISE .IMG or .JP2 file.
    """

    def __init__(self, img_path: str | os.PathLike = None, img=None):
        self._temp_file_path = None  # Keep track of temp file to delete later

        if img_path:
            with rasterio.open(img_path) as src:
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

            self.img_path = img_path
            # Extracted file name without extension regardless of format (.IMG, .JP2, .jp2, etc.)
            self.file_name = os.path.splitext(os.path.basename(img_path))[0]
            self.metadata = self._get_metadata()

        else:
            data = np.array(img)
            self.numpy_image = data

    def __del__(self):
        """
        Cleanup: Ensure the temporary file is deleted when the object is destroyed.
        """
        # Close the memmap reference if possible (numpy handles this usually)
        if hasattr(self, 'numpy_image') and isinstance(self.numpy_image, np.memmap):
            self.numpy_image._mmap.close()
            del self.numpy_image

        # Delete the actual file from disk
        if self._temp_file_path and os.path.exists(self._temp_file_path):
            try:
                os.remove(self._temp_file_path)
            except PermissionError:
                pass  # Windows sometimes holds locks longer than expected

    def get_portion_of_map(self, size, max_percentage_inf=0):
        # Extracts a size x size portion of the image, avoiding too many np.inf
        img_height, img_width = self.numpy_image.shape[:2]

        while True:
            # pick random top-left corner
            x = np.random.randint(0, img_width - size + 1)
            y = np.random.randint(0, img_height - size + 1)

            # extract portion
            image_subset = np.array(self.numpy_image[y:y + size, x:x + size])

            # count infinities
            num_inf = np.sum(np.isinf(image_subset))
            if num_inf <= max_percentage_inf * (size * size):
                break

        # return image portion and its coordinates as (row,column)=(y,x)
        return image_subset, (y, x)

    def apply_nodata_patch(self, y, x, size):
        """Given a coordinate (upper left corner) and a patch size, apply a no-data patch to the dtm to obscure that area"""
        black_fill = np.full(shape=(size, size), fill_value=np.inf)
        self.numpy_image[y:y + size, x:x + size] = black_fill

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
        # Example: DTEEC_016460_2230_016170_2230_G01
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
        # Example: ESP_043599_1650_RED
        elif len(parts) == 4:
            phase_or_type, orbit_number, lat_band, band_color = parts

            return {
                "product_type": "Ortho Image",
                "phase": phase_or_type,  # e.g., ESP or PSP
                "orbit_number": orbit_number,
                "latitude_band": lat_band,
                "band_color": band_color  # e.g., RED or COLOR
            }

        # Fallback for unexpected file name structures
        else:
            return {
                "product_type": unk,
                "raw_filename": self.file_name
            }
