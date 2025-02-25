import os
import tifffile as tiff
import numpy as np

# Paths (Keeping original folders untouched)
image_folder = "omdennavf/cropped_images/"
mask_folder = "omdennavf/cropped_masks/"

# Clean output directories
clean_image_folder = "omdennavf/clean_cropped_images/"
clean_mask_folder = "omdennavf/clean_cropped_masks/"

# Ensure clean folders exist
os.makedirs(clean_image_folder, exist_ok=True)
os.makedirs(clean_mask_folder, exist_ok=True)

# Function to check if an image is valid
def is_valid_image(image, expected_channels, min_content_threshold=0.01):
    """Check if an image is valid and has correct shape."""
    if image is None:
        return False
    if len(image.shape) != 3:
        return False
    if image.shape[0] != expected_channels:
        return False
    if np.isnan(image).any():  # Check for NaN values
        return False
    if np.max(image) == 0:  # Completely black image
        return False
    if np.mean(image) < min_content_threshold:  # Check for low content in image
        return False
    return True

def is_valid_mask(mask, expected_channels, min_content_threshold=0.01):
    """Check if the mask is valid."""
    if mask is None:
        return False
    if len(mask.shape) != 3:
        return False
    if mask.shape[0] != expected_channels:
        return False
    if np.isnan(mask).any():  # Check for NaN values
        return False
    if np.max(mask) == 0:  # Completely black mask
        return False
    if np.mean(mask) < min_content_threshold:  # Check for low content in mask
        return False
    return True

# Process images and masks
for filename in os.listdir(image_folder):
    if filename.endswith(".tif"):
        # Construct mask filenames
        mask_filename_1 = filename.replace(".tif", "_fractional_mask.tif")
        mask_filename_2 = filename.replace(".tif", "_fractional_mask.term.tif")

        image_path = os.path.join(image_folder, filename)
        mask_path_1 = os.path.join(mask_folder, mask_filename_1)
        mask_path_2 = os.path.join(mask_folder, mask_filename_2)

        # Determine which mask exists
        if os.path.exists(mask_path_1):
            mask_path = mask_path_1
        elif os.path.exists(mask_path_2):
            mask_path = mask_path_2
        else:
            print(f"⚠️ Mask not found for {filename}, skipping...")
            continue

        # Read image and mask
        try:
            image = tiff.imread(image_path)
            mask = tiff.imread(mask_path)
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            continue

        # Ensure correct format (C, H, W)
        if len(image.shape) == 3 and image.shape[-1] == 19:
            image = np.transpose(image, (2, 0, 1))  # Convert (H, W, C) → (C, H, W)
        if len(mask.shape) == 3 and mask.shape[-1] == 5:
            mask = np.transpose(mask, (2, 0, 1))  # Convert (H, W, C) → (C, H, W)

        # Validate image and mask
        if not is_valid_image(image, 19):
            print(f"⚠️ Skipping {filename} - invalid image format, corrupted, or lacks content.")
            continue
        if not is_valid_mask(mask, 5):
            print(f"⚠️ Skipping {mask_filename_1} - invalid mask format, corrupted, or lacks content.")
            continue

        # Replace NaNs in masks with 0
        mask = np.nan_to_num(mask)

        # Save cleaned images
        tiff.imwrite(os.path.join(clean_image_folder, filename), image)
        tiff.imwrite(os.path.join(clean_mask_folder, os.path.basename(mask_path)), mask)

print("✅ Cleaning process completed successfully!")
