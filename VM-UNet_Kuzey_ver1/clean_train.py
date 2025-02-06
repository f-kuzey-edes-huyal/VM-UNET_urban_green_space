import os
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm
import shutil

# Define paths
base_dir = "/mnt/d/VM-UNet/data/omdena"
train_images_dir = os.path.join(base_dir, "train/images")
train_masks_dir = os.path.join(base_dir, "train/masks")
val_images_dir = os.path.join(base_dir, "val/images")
val_masks_dir = os.path.join(base_dir, "val/masks")

# Output cleaned directories
clean_train_images_dir = os.path.join(base_dir, "cleaned/train/images")
clean_train_masks_dir = os.path.join(base_dir, "cleaned/train/masks")
clean_val_images_dir = os.path.join(base_dir, "cleaned/val/images")
clean_val_masks_dir = os.path.join(base_dir, "cleaned/val/masks")

# Create output directories if they don’t exist
os.makedirs(clean_train_images_dir, exist_ok=True)
os.makedirs(clean_train_masks_dir, exist_ok=True)
os.makedirs(clean_val_images_dir, exist_ok=True)
os.makedirs(clean_val_masks_dir, exist_ok=True)

# Function to clean images and keep only relevant channels
def clean_images_and_masks(image_dir, mask_dir, output_image_dir, output_mask_dir, selected_channels):
    for filename in tqdm(os.listdir(image_dir), desc=f"Processing {image_dir}"):
        if filename.endswith(".tif"):
            image_path = os.path.join(image_dir, filename)

            # Construct corresponding mask filename
            mask_filename = filename.replace(".tif", "_fractional_mask.tif")
            mask_path = os.path.join(mask_dir, mask_filename)
            
            # If the mask does not exist, skip and do not copy the image
            if not os.path.exists(mask_path):
                print(f"Skipping {filename} - No corresponding mask found.")
                continue

            try:
                # Open image with rasterio
                with rasterio.open(image_path) as src:
                    image = src.read()  # Shape: (num_channels, height, width)
                
                # Check if image has expected number of channels
                if image.shape[0] < max(selected_channels) + 1:
                    print(f"Skipping {filename} - Not enough channels")
                    continue

                # Select only the required 12 channels
                image = image[selected_channels, :, :]

                # Save the cleaned image
                cleaned_image_path = os.path.join(output_image_dir, filename)
                with rasterio.open(
                    cleaned_image_path,
                    'w',
                    driver='GTiff',
                    height=image.shape[1],
                    width=image.shape[2],
                    count=len(selected_channels),
                    dtype=image.dtype,
                    crs=src.crs,
                    transform=src.transform
                ) as dst:
                    dst.write(image)

                # Copy the corresponding mask
                shutil.copy(mask_path, os.path.join(output_mask_dir, mask_filename))

            except (RasterioIOError, ValueError, OSError) as e:
                print(f"Error processing {filename}: {e}")
                continue

# Define the 12 important channels (indexing starts at 0)
important_channels = list(range(12))  # Modify if needed

# Clean train and validation sets
clean_images_and_masks(train_images_dir, train_masks_dir, clean_train_images_dir, clean_train_masks_dir, important_channels)
clean_images_and_masks(val_images_dir, val_masks_dir, clean_val_images_dir, clean_val_masks_dir, important_channels)

print("Data cleaning complete! 🎉")
