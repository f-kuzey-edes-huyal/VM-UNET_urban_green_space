import os
import numpy as np
import tifffile as tiff
from pathlib import Path

# Define paths
image_folder = "/mnt/c/Users/Kuzey/omdene_images_masks/images/VBWVA_8R/"  # Change this to your image folder
mask_folder = "/mnt/c/Users/Kuzey/omdene_images_masks/masks/VBWVA_8R_masks/"    # Change this to your mask folder
output_image_folder = "/mnt/d/VM-UNet/data/new3/train/images/"
output_mask_folder = "/mnt/d/VM-UNet/data/new3/train/masks/"

# Ensure output folders exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

def center_crop(img, crop_size=128):
    """Crops the center region of an image/mask to crop_size x crop_size."""
    h, w = img.shape[1], img.shape[2]  # Get original height & width (not channels)
    start_x = max(0, (w - crop_size) // 2)
    start_y = max(0, (h - crop_size) // 2)
    return img[:, start_y:start_y+crop_size, start_x:start_x+crop_size]

# Process images and masks
for img_file in sorted(Path(image_folder).glob("*.tif")):
    # Mask file should have the same base name, but with the _fractional_masks suffix
    mask_file = Path(mask_folder) / (img_file.stem + "_fractional_mask.tif")

    if mask_file.exists():
        # Load image and mask
        image = tiff.imread(str(img_file))  # Shape: (19, H, W)
        mask = tiff.imread(str(mask_file))  # Shape: (5, H, W)

        # Center crop
        cropped_image = center_crop(image)
        cropped_mask = center_crop(mask)

        # Save cropped files
        tiff.imwrite(str(Path(output_mask_folder) / mask_file.name), cropped_mask)
        tiff.imwrite(str(Path(output_image_folder) / img_file.name), cropped_image)

        print(f"Cropped and saved: {img_file.name} and {mask_file.name}")
    else:
        print(f"Skipping {img_file.name}, corresponding mask not found at: {mask_file}")

print("Processing complete!")
