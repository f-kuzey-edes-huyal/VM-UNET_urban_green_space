import os
import rasterio
import numpy as np
import torch
from pathlib import Path
from PIL import Image  # Added import for Image
from tqdm import tqdm  # Progress bar

def process_geotiff_file(file_path, target_size=(128, 128), channels_to_use=19):
    """Process a single GeoTIFF file and return a numpy array of shape [C, H, W]."""
    with rasterio.open(file_path) as src:
        # Read only the required number of channels
        image = src.read()[:channels_to_use].astype(np.float32)  # Shape: [C, H, W]

        # Resize each channel separately (avoid high memory usage)
        resized_channels = np.zeros((channels_to_use, target_size[0], target_size[1]), dtype=np.float32)
        for c in range(channels_to_use):
            resized_channels[c] = np.array(
                Image.fromarray(image[c]).resize(target_size, Image.Resampling.LANCZOS)
            )

        return resized_channels  # Shape: [C, H, W]

def compute_streaming_mean_std(folder_paths, target_size=(128, 128), channels_to_use=19):
    """Compute running mean and std per channel to reduce memory usage."""
    mean = np.zeros(channels_to_use, dtype=np.float64)
    var = np.zeros(channels_to_use, dtype=np.float64)
    count = 0

    for folder_path in folder_paths:
        for filename in tqdm(os.listdir(folder_path), desc=f"Processing {folder_path}", unit="file"):
            if filename.endswith(".tif") or filename.endswith(".tiff"):
                file_path = os.path.join(folder_path, filename)
                image = process_geotiff_file(file_path, target_size, channels_to_use)  # [C, H, W]

                # Compute mean and variance using Welford's algorithm (numerically stable)
                for c in range(channels_to_use):
                    pixels = image[c].flatten()  # Convert [H, W] to [H*W]
                    n = len(pixels)

                    # Update running mean and variance
                    delta = pixels - mean[c]
                    mean[c] += delta.sum() / (count + n)
                    var[c] += (delta * (pixels - mean[c])).sum()
                    count += n

    # Finalize standard deviation calculation
    std = np.sqrt(var / count)

    return mean, std

# Define data folders
data_folder = Path("/mnt/c/Users/Kuzey/omdene_images_masks/images")
folders = [
    data_folder / "SLMO_8R_1",
    data_folder / "SLMO_9R_2",
    data_folder / "SLMO_9R_3",
    data_folder / "VBWVA_8R"
]

# Compute efficient mean and std
mean_channels, std_channels = compute_streaming_mean_std(folders, target_size=(128, 128), channels_to_use=19)

print("Mean of each channel:", mean_channels)
print("Standard Deviation of each channel:", std_channels)
