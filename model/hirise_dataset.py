import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class Hirise_Dataset(Dataset):
    def __init__(
            self,
            annotations_dataframe: pd.DataFrame,
            root_dir: str = "",  # Add root_dir (e.g. data_path or absolute project directory)
            transform=None,
            target_transform=None,
            generate_synthetic_thermal: bool = True,
            sequence_length: int = 3
    ):
        self.root_dir = root_dir
        self.img_dirs = annotations_dataframe["img_path"].tolist()
        self.img_names = annotations_dataframe["image_name"].tolist()
        self.img_labels = annotations_dataframe["category_id"].tolist()

        self.transform = transform
        self.target_transform = target_transform
        self.generate_synthetic_thermal = generate_synthetic_thermal
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int):
        # Resolve full path using absolute root_dir
        img_path = os.path.abspath(
            os.path.join(self.root_dir, self.img_dirs[idx], self.img_names[idx])
        )

        raw_img = Image.open(img_path)
        uint8_np = self.scale_hirise_to_uint8(np.array(raw_img))
        image = Image.fromarray(uint8_np)
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        if self.generate_synthetic_thermal:
            h, w = (image.shape[-2:] if isinstance(image, torch.Tensor) else (image.size[1], image.size[0]))
            thermal = torch.randn(self.sequence_length, 1, h, w, dtype=torch.float32)
            return image, thermal, label

        return image, label

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