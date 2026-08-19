import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from PIL import Image
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from hirise_dtm import HiriseDTM
from utils import *

# Ensure output directory exists
output_dir = "data/plain_terrain_dataset"
os.makedirs(output_dir, exist_ok=True)

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


# Divide the sum of means and std by the total number of images
mean /= len(deep_landforms_metadata_imgs)
std /= len(deep_landforms_metadata_imgs)


# Find smallest and biggest images based on total pixel count
smallest_shape = min(shapes, key=lambda s: s[0] * s[1])
biggest_shape = max(shapes, key=lambda s: s[0] * s[1])

######## Probability distribution of image size over dataset ########
size_counts = {shape[0]: count for shape, count in shapes.items()}

sizes = list(size_counts.keys())
counts = list(size_counts.values())
sum_counts = sum(counts)

probabilities = [count/sum_counts for count in counts]
size_probability_dist = {size: prob for size, prob in zip(sizes, probabilities)}


print("Mean and standard deviation of image channels:")
print(f"Mean: {mean}")
print(f"Std: {std}")

print("\nImage dimensions:")
print(f"Smallest image: {smallest_shape} "
      f"({smallest_shape[0] * smallest_shape[1]:,} pixels)")

print(f"Biggest image: {biggest_shape} "
      f"({biggest_shape[0] * biggest_shape[1]:,} pixels)")

print(f"Number of different image sizes: {len(shapes)}")





for jp2_image in hirise_imgs:
    if stop_pipeline:
        break

    filepath = f"data/DTMs/{jp2_image}"
    # HiriseDTM already uses a memmap on disk, so apply_black_patch/apply_nodata_patch
    # modifies the backing binary temporary/mapped file in place.
    print("Loading DTM image...")
    dtm_file = HiriseDTM(filepath)
    print("DTM image loaded")

    samples_collected = 0
    target_samples = 100

    print(f"\n--- Processing {jp2_image} ---")
    print("Controls:\n  [y / Enter] = Approve and Save\n  [n / Space] = Reject\n  [q / Esc]   = Save state and Quit execution\n")

    plt.ion()  # Turn on interactive mode so plt.show() doesn't block input

    while samples_collected < target_samples and not stop_pipeline:
        sample_size = sample_image_size(size_probability_dist)
        image, (y, x) = dtm_file.get_portion_of_map(sample_size)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap="terrain")
        ax.set_title(
            f"{jp2_image}\nSample {samples_collected + 1}/{target_samples} (Size: {sample_size}px)"
        )
        plt.show(block=False)
        plt.pause(0.1)  # Force render the plot window

        # Prompt user directly in terminal console
        user_choice = input(
            f"Sample {samples_collected + 1}/{target_samples} | Accept region? [y = Accept, n = Reject, q = Save & Quit]: "
        ).strip().lower()

        plt.close(fig)

        if user_choice == 'q':
            print("Stopping pipeline requested. Flushing DTM state...")
            if hasattr(dtm_file.numpy_image, 'flush'):
                dtm_file.numpy_image.flush()
            stop_pipeline = True
            break


        elif user_choice == 'y':
            # 1. Prepara il nome del file con estensione .JP2
            crop_filename = f"{dtm_file.file_name}_y{y}_x{x}_s{sample_size}.JP2"
            crop_path = os.path.join(output_dir, crop_filename)

            # Normalizzazione dell'array di elevazione (float) a 8-bit (0-255) per la conversione in immagine
            # Se ci sono valori inf/nan nel crop li gestiamo con nanmin/nanmax
            valid_mask = np.isfinite(image)
            if np.any(valid_mask):
                min_val = np.min(image[valid_mask])
                max_val = np.max(image[valid_mask])
                if max_val > min_val:
                    # Scaling a 0-255
                    norm_img = ((image - min_val) / (max_val - min_val) * 255.0)
                else:
                    norm_img = np.zeros_like(image)
            else:
                norm_img = np.zeros_like(image)
            # Sostituiamo eventuali inf/nan rimasti con 0
            norm_img = np.nan_to_num(norm_img, nan=0.0, posinf=0.0, neginf=0.0)
            img_uint8 = norm_img.astype(np.uint8)
            # Salvataggio in formato JP2 tramite PIL
            pil_img = Image.fromarray(img_uint8)
            pil_img.save(crop_path, format="JPEG2000")
            # 2. Log annotation record
            new_row = {
                "file_name": dtm_file.file_name,
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

            # 3. Apply nodata patch to DTM to obscure selected terrain
            dtm_file.apply_nodata_patch(y, x, sample_size)
            samples_collected += 1
            print(f"Approved and saved as JP2 ({samples_collected}/{target_samples})\n")

        else:
            print("Rejected sample. Picking another region...\n")