import os
import tifffile as tiff
import numpy as np

# Define paths
image_folder = "/mnt/c/Users/Kuzey/omdene_images_masks/images/VBWVA_8R/"
mask_folder = "/mnt/c/Users/Kuzey/omdene_images_masks/masks/VBWVA_8R_masks/"
output_image_folder = "omdennavf/cropped_images/"
output_mask_folder = "omdennavf/cropped_masks/"

# Ensure output directories exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)


# Center crop function
def center_crop(image, crop_size=128):
    h, w = image.shape[1:3]  # Get spatial dimensions
    center_x, center_y = w // 2, h // 2
    x1 = center_x - crop_size // 2
    y1 = center_y - crop_size // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    return image[:, y1:y2, x1:x2]  # Keep all channels


# Process images and masks
for filename in os.listdir(image_folder):
    if filename.endswith(".tif"):
        # Construct expected mask filename
        mask_filename = filename.replace(".tif", "_fractional_mask.tif")
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"Mask not found for {filename}, skipping...")
            continue

        # Read image and mask (preserve all channels)
        image = tiff.imread(image_path)  # Can be (H, W, C) or (C, H, W)
        mask = tiff.imread(mask_path)  # Can be (H, W, C) or (C, H, W)

        # Ensure image and mask have expected channel sizes
        if image.shape[-1] == 19 and len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))  # Convert (H, W, C) → (C, H, W)
        elif image.shape[0] != 19:
            print(f"Skipping {filename} - unexpected channel size in image: {image.shape}")
            continue

        if mask.shape[-1] == 5 and len(mask.shape) == 3:
            mask = np.transpose(mask, (2, 0, 1))  # Convert (H, W, C) → (C, H, W)
        elif mask.shape[0] != 5:
            print(f"Skipping {mask_filename} - unexpected channel size in mask: {mask.shape}")
            continue
        
        # Ensure images are large enough for cropping
        if image.shape[1] < 128 or image.shape[2] < 128:
            print(f"Skipping {filename} - too small for cropping.")
            continue
        
        # Apply center crop
        cropped_image = center_crop(image)
        cropped_mask = center_crop(mask)

        # Save the cropped images
        tiff.imwrite(os.path.join(output_image_folder, filename), cropped_image)
        tiff.imwrite(os.path.join(output_mask_folder, mask_filename), cropped_mask)

print("Center cropping completed successfully!")
