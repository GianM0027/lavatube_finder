import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from PIL import Image
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from hirise_dtm import HiriseDTM
from utils import *

# Ensure output directory exists
output_dir = "data/plain_terrain_dataset"
os.makedirs(output_dir, exist_ok=True)

# Path to store patch coordinates persistent state
patches_state_file = os.path.join(output_dir, "dtm_patches_state.json")

# Load existing patch states if file exists, else create empty dictionary
if os.path.exists(patches_state_file):
    with open(patches_state_file, "r") as f:
        dtm_patches_state = json.load(f)
else:
    dtm_patches_state = {}

data_path = "data"
deeplandforms_path = os.path.join(data_path, "DeepLandforms_dataset", "dataset.json")

with open(deeplandforms_path, 'r') as json_file:
    deep_landforms_metadata = json.load(json_file)
    deep_landforms_metadata_imgs = pd.DataFrame(deep_landforms_metadata["images"])
    deep_landforms_metadata_annotations = pd.DataFrame(deep_landforms_metadata["annotations"])
    deep_landforms_metadata_categories = pd.DataFrame(deep_landforms_metadata["categories"])

plain_terrain_annotations = pd.DataFrame(columns=deep_landforms_metadata_annotations.columns)

hirise_imgs = [
    "ESP_043599_1650_RED.JP2", "ESP_087423_2360_RED.JP2", "ESP_087440_2565_RED.JP2",
    "ESP_087443_2650_RED.JP2", "ESP_087409_2420_RED.JP2", "ESP_087422_2570_RED.JP2"
]

stop_pipeline = False

# Initialize variables to accumulate mean and std
mean = torch.zeros(1)
std = torch.zeros(1)
num_pixels = 0

# Dictionary to keep track of image shapes
shapes = {}

# Loop through the global dataset
for _, row in tqdm(
        deep_landforms_metadata_imgs.iterrows(),
        desc="Collecting image dimensions and computing statistics",
        total=len(deep_landforms_metadata_imgs)
):
    img = Image.open(os.path.join("data", f'DeepLandforms_dataset/{row["file_name"]}'))

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    img = transform(img).float()
    channels, height, width = img.shape

    # Update shape occurrences
    shape = (width, height)
    shapes[shape] = shapes.get(shape, 0) + 1

    # Compute the number of pixels per image
    num_pixels += height * width

    # Sum the mean values of each channel
    mean += img.view(channels, -1).mean(dim=1)

    # Sum the standard deviation of each channel
    std += img.view(channels, -1).std(dim=1)

mean /= len(deep_landforms_metadata_imgs)
std /= len(deep_landforms_metadata_imgs)

smallest_shape = min(shapes, key=lambda s: s[0] * s[1])
biggest_shape = max(shapes, key=lambda s: s[0] * s[1])

size_counts = {shape[0]: count for shape, count in shapes.items()}
sizes = list(size_counts.keys())
counts = list(size_counts.values())
sum_counts = sum(counts)

probabilities = [count / sum_counts for count in counts]
size_probability_dist = {size: prob for size, prob in zip(sizes, probabilities)}

print("Mean and standard deviation of image channels:")
print(f"Mean: {mean}")
print(f"Std: {std}\n")

plt.ion()  # Non-blocking plot mode

for jp2_image in hirise_imgs:
    if stop_pipeline:
        break

    filepath = f"data/DTMs/{jp2_image}"
    print(f"\nLoading DTM image: {jp2_image}...")
    dtm_file = HiriseDTM(filepath)
    dtm_name = dtm_file.file_name

    # Check if there are saved patches for this DTM file
    if dtm_name in dtm_patches_state and len(dtm_patches_state[dtm_name]) > 0:
        saved_patches = dtm_patches_state[dtm_name]
        load_choice = input(
            f"Found {len(saved_patches)} saved patches for {dtm_name}. Restore them? [y/n]: "
        ).strip().lower()

        if load_choice == 'y':
            print("Applying saved patches to DTM...")
            for patch in saved_patches:
                dtm_file.apply_black_patch(patch["y"], patch["x"], patch["size"])
            print("Patches restored successfully.")

    samples_collected = 0
    target_samples = 100

    print(f"--- Processing {jp2_image} ---")

    while samples_collected < target_samples and not stop_pipeline:
        sample_size = sample_image_size(size_probability_dist)
        image, (y, x) = dtm_file.get_portion_of_map(sample_size)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap="terrain")
        ax.set_title(
            f"{jp2_image}\nSample {samples_collected + 1}/{target_samples} (Size: {sample_size}px)"
        )
        plt.show(block=False)
        plt.pause(0.1)

        user_choice = input(
            f"Sample {samples_collected + 1}/{target_samples} | Accept region? [y = Accept, n = Reject, q = Save & Quit]: "
        ).strip().lower()

        plt.close(fig)

        if user_choice == 'q':
            print("Stopping pipeline requested. Saving patch states...")
            if hasattr(dtm_file.numpy_image, 'flush'):
                dtm_file.numpy_image.flush()
            stop_pipeline = True
            break

        elif user_choice == 'y':
            # 1. Save crop image as GeoTIFF
            crop_filename = f"{dtm_name}_y{y}_x{x}_s{sample_size}.tif"
            crop_path = os.path.join(output_dir, crop_filename)

            with rasterio.open(
                    crop_path,
                    'w',
                    driver='GTiff',
                    height=image.shape[0],
                    width=image.shape[1],
                    count=1,
                    dtype=image.dtype
            ) as dst:
                dst.write(image, 1)

            # 2. Log annotation record
            new_row = {
                "file_name": dtm_name,
                "crop_file": crop_filename,
                "y": y,
                "x": x,
                "size": sample_size,
                "label": "plain_terrain"
            }
            plain_terrain_annotations = pd.concat(
                [plain_terrain_annotations, pd.DataFrame([new_row])],
                ignore_index=True
            )

            # 3. Apply nodata patch to the DTM instance
            dtm_file.apply_nodata_patch(y, x, sample_size)

            # 4. Save patch coordinate to global tracking state
            if dtm_name not in dtm_patches_state:
                dtm_patches_state[dtm_name] = []

            dtm_patches_state[dtm_name].append({"y": int(y), "x": int(x), "size": int(sample_size)})

            # Update JSON state file on disk after every approved patch
            with open(patches_state_file, "w") as f:
                json.dump(dtm_patches_state, f, indent=4)

            samples_collected += 1
            print(f"Approved and saved ({samples_collected}/{target_samples})\n")

        else:
            print("Rejected sample. Picking another region...\n")

print(f"\nExecution ended. Collected {len(plain_terrain_annotations)} samples in total.")