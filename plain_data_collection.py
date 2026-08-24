import os
import re
import sys
import json
import warnings
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
from rasterio.windows import Window

# Ensure output directory exists
output_dir = "data/plain_terrain_dataset"
os.makedirs(output_dir, exist_ok=True)

# Path to store patch coordinates persistent state
patches_state_file = os.path.join(output_dir, "dtm_patches_state.json")
annotations_csv = os.path.join(output_dir, "plain_terrain_annotations.csv")

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

# Load existing annotations CSV if it exists so previous labeling runs are never overwritten!
if os.path.exists(annotations_csv):
    plain_terrain_annotations = pd.read_csv(annotations_csv)
    print(f"Successfully loaded {len(plain_terrain_annotations)} existing annotations from previous runs.")

    # Check highest existing image_id to continue seamlessly from it
    if not plain_terrain_annotations.empty and 'image_id' in plain_terrain_annotations.columns:
        max_id = plain_terrain_annotations['image_id'].max()
        next_image_id = int(max_id) + 1 if pd.notna(max_id) and max_id >= 1847 else 1847
    else:
        next_image_id = 1847
else:
    plain_terrain_annotations = pd.DataFrame(columns=deep_landforms_metadata_annotations.columns)
    next_image_id = 1847

hirise_imgs = [
    #"ESP_011287_2165_RED.JP2",
    #"ESP_011293_1710_RED.JP2",
    #"ESP_011325_1845_RED.JP2",
    "ESP_011335_1005_RED.JP2",
    "ESP_043599_1650_RED.JP2",
    "ESP_087433_2545_RED.JP2"
]

stop_pipeline = False

# Compute global stats over DeepLandforms dataset
mean = torch.zeros(1)
std = torch.zeros(1)
num_pixels = 0
shapes = {}

for _, row in tqdm(
        deep_landforms_metadata_imgs.iterrows(),
        desc="Collecting image dimensions and computing statistics",
        total=len(deep_landforms_metadata_imgs)
):
    img = Image.open(os.path.join("data", f'DeepLandforms_dataset/{row["file_name"]}'))
    transform = transforms.Compose([transforms.PILToTensor()])
    img = transform(img).float()
    channels, height, width = img.shape

    shape = (width, height)
    shapes[shape] = shapes.get(shape, 0) + 1
    num_pixels += height * width
    mean += img.view(channels, -1).mean(dim=1)
    std += img.view(channels, -1).std(dim=1)

mean /= len(deep_landforms_metadata_imgs)
std /= len(deep_landforms_metadata_imgs)

size_counts = {shape[0]: count for shape, count in shapes.items()}
sizes = list(size_counts.keys())
counts = list(size_counts.values())
sum_counts = sum(counts)

probabilities = [count / sum_counts for count in counts]
size_probability_dist = {size: prob for size, prob in zip(sizes, probabilities)}

plt.ion()  # Non-blocking plot mode

for jp2_image in hirise_imgs:
    if stop_pipeline:
        break

    filepath = f"data/DTMs/{jp2_image}"

    # Peek at metadata / LBL file to show resolution before heavy loading
    lbl_path = os.path.splitext(filepath)[0] + ".LBL"
    resolution = "Unknown"
    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            match = re.search(r"MAP_SCALE\s*=\s*(-?\d+\.?\d*)", f.read())
            if match:
                resolution = match.group(1)

    print(f"\n----------------------------------------")
    print(f"Image Found: {jp2_image}")
    print(f"Image Resolution: {resolution} meters/pixel")
    print(f"----------------------------------------")

    proceed_choice = input(
        f"Do you want to proceed with {jp2_image}? [y = Proceed, n = Skip, q = Quit pipeline]: ").strip().lower()

    if proceed_choice == 'q':
        print("Stopping pipeline requested.")
        stop_pipeline = True
        break
    elif proceed_choice != 'y':
        print(f"Skipping image {jp2_image}.\n")
        continue

    print(f"\nLoading DTM image into memory: {jp2_image}...")

    # Extract CRS and affine transform metadata directly from rasterio source
    with rasterio.open(filepath) as src:
        dtm_crs = src.crs
        dtm_transform = src.transform

    dtm_file = HiriseDTM(filepath)
    dtm_name = dtm_file.file_name

    # Total pixels available in this specific DTM
    total_map_pixels = dtm_file.numpy_image.shape[0] * dtm_file.numpy_image.shape[1]

    # Check if there are saved patches for this DTM file
    if dtm_name in dtm_patches_state and len(dtm_patches_state[dtm_name]) > 0:
        saved_patches = dtm_patches_state[dtm_name]
        load_choice = input(
            f"Found {len(saved_patches)} saved patches for {dtm_name}. Restore them? [y/n]: "
        ).strip().lower()

        if load_choice == 'y':
            print("Applying saved patches to DTM...")
            for patch in saved_patches:
                dtm_file.apply_nodata_patch(patch["y"], patch["x"], patch["size"])
            print("Patches restored successfully.")

    samples_collected = 0
    target_samples = 100

    print(f"--- Processing {jp2_image} ---")

    while samples_collected < target_samples and not stop_pipeline:
        # Calculate current map coverage percentage based on saved state
        approved_pixels = sum(p["size"] ** 2 for p in dtm_patches_state.get(dtm_name, []))
        coverage_pct = (approved_pixels / total_map_pixels) * 100

        sample_size = sample_image_size(size_probability_dist)
        image, (y, x) = dtm_file.get_portion_of_map(sample_size, max_percentage_invalid=0.1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"{jp2_image} | Coverage: {coverage_pct:.2f}%\n"
            f"Sample {samples_collected + 1}/{target_samples} (Size: {sample_size}px)"
        )
        plt.show(block=False)
        plt.pause(0.1)

        user_choice = input(
            f"[{coverage_pct:.2f}% map used] Sample {samples_collected + 1}/{target_samples} | "
            f"Accept region? [y = Accept, n = Reject, u = Undo last, q = Save & Quit]: "
        ).strip().lower()

        plt.close(fig)

        if user_choice == 'q':
            print("Stopping pipeline requested. Saving patch states...")
            if hasattr(dtm_file.numpy_image, 'flush'):
                dtm_file.numpy_image.flush()
            stop_pipeline = True
            break

        elif user_choice == 'u':
            # Undo last accepted sample action
            if samples_collected > 0 and dtm_name in dtm_patches_state and len(dtm_patches_state[dtm_name]) > 0:
                last_patch = dtm_patches_state[dtm_name].pop()
                y_u, x_u, size_u = last_patch["y"], last_patch["x"], last_patch["size"]

                # Remove saved crop file
                crop_filename = f"{dtm_name}_y{y_u}_x{x_u}_s{size_u}.tiff"
                crop_path = os.path.join(output_dir, crop_filename)
                if os.path.exists(crop_path):
                    os.remove(crop_path)

                # Restore original pixels in memory map
                dtm_file.undo_nodata_patch(y_u, x_u, size_u)

                # Remove last row from annotations dataframe matching this crop file and update CSV
                if len(plain_terrain_annotations) > 0 and crop_filename in plain_terrain_annotations[
                    "image_name"].values:
                    plain_terrain_annotations = plain_terrain_annotations[
                        plain_terrain_annotations["image_name"] != crop_filename].reset_index(drop=True)
                elif len(plain_terrain_annotations) > 0:
                    plain_terrain_annotations = plain_terrain_annotations.iloc[:-1].reset_index(drop=True)

                plain_terrain_annotations.to_csv(annotations_csv, index=False)

                # Re-evaluate next_image_id based on current dataframe highest ID
                if not plain_terrain_annotations.empty and 'image_id' in plain_terrain_annotations.columns:
                    max_id = plain_terrain_annotations['image_id'].max()
                    next_image_id = int(max_id) + 1 if pd.notna(max_id) and max_id >= 1847 else 1847
                else:
                    next_image_id = 1847

                # Update JSON state file
                with open(patches_state_file, "w") as f:
                    json.dump(dtm_patches_state, f, indent=4)

                samples_collected -= 1
                print(f"Undo successful! Reverted last sample. Current count: {samples_collected}/{target_samples}\n")
            else:
                print("Nothing to undo for this image session.\n")

        elif user_choice == 'y':
            # Assign current image_id and id, then increment for next
            current_image_id = next_image_id
            next_image_id += 1

            # 1. Retrieve Projected Coordinates (Meters) directly via class method
            x_min_proj, y_max_proj = dtm_file.get_pixel_coordinate(y, x)
            x_center_proj, y_center_proj = dtm_file.get_pixel_coordinate(
                y + sample_size // 2, x + sample_size // 2
            )

            # 2. Retrieve Global Mars Latitude & Longitude directly via class method
            lat_topleft, lon_topleft = dtm_file.get_lat_lon(y, x)
            lat_center, lon_center = dtm_file.get_lat_lon(
                y + sample_size // 2, x + sample_size // 2
            )

            # 3. Create sub-window transform for GeoTIFF export
            crop_window = rasterio.windows.Window(
                col_off=x, row_off=y, width=sample_size, height=sample_size
            )
            crop_transform = rasterio.windows.transform(crop_window, dtm_transform)

            # 4. Save GeoTIFF file retaining spatial CRS & sub-transform
            crop_filename = f"{dtm_name}_y{y}_x{x}_s{sample_size}.tif"
            crop_path = os.path.join(output_dir, crop_filename)
            with rasterio.open(
                    crop_path,
                    'w',
                    driver='GTiff',
                    height=image.shape[0],
                    width=image.shape[1],
                    count=1,
                    dtype=image.dtype,
                    crs=dtm_crs,
                    transform=crop_transform
            ) as dst:
                dst.write(image, 1)

            # 5. Log metadata record including custom COCO fields
            new_row = {
                "file_name": dtm_name,
                "image_name": crop_filename,
                "pixel_y": y,
                "pixel_x": x,
                "size": sample_size,
                "lat_center": lat_center,
                "lon_center": lon_center,
                "lat_topleft": lat_topleft,
                "lon_topleft": lon_topleft,
                "proj_x_center_m": x_center_proj,
                "proj_y_center_m": y_center_proj,
                "crs": str(dtm_crs),
                "label": "plain_terrain",
                "image_id": current_image_id,
                "id": current_image_id,
                "iscrowd": 0,
                "bbox": [],
                "segmentation": [],
                "category_id": 2,
                "area": 0
            }

            plain_terrain_annotations = pd.concat(
                [plain_terrain_annotations, pd.DataFrame([new_row])],
                ignore_index=True
            )

            # Update annotation CSV safely keeping past records
            plain_terrain_annotations.to_csv(annotations_csv, index=False)

            # 6. Apply nodata patch to DTM memory map (fills with zeros to block re-sampling)
            dtm_file.apply_nodata_patch(y, x, sample_size)

            # 7. Record state in JSON
            if dtm_name not in dtm_patches_state:
                dtm_patches_state[dtm_name] = []

            dtm_patches_state[dtm_name].append({
                "y": int(y),
                "x": int(x),
                "size": int(sample_size),
                "lat_center": float(lat_center),
                "lon_center": float(lon_center)
            })

            with open(patches_state_file, "w") as f:
                json.dump(dtm_patches_state, f, indent=4)

            samples_collected += 1

            # Recalculate coverage for print statement
            approved_pixels = sum(p["size"] ** 2 for p in dtm_patches_state[dtm_name])
            coverage_pct = (approved_pixels / total_map_pixels) * 100

            print(
                f"Approved ({samples_collected}/{target_samples}) | Image ID: {current_image_id} | "
                f"Map Coverage: {coverage_pct:.2f}% | "
                f"Mars Center: ({lat_center:.4f}° N, {lon_center:.4f}° E)\n"
            )

        else:
            print("Rejected sample. Picking another region...\n")

print(f"\nExecution ended. Total accumulated annotations in dataset: {len(plain_terrain_annotations)}")