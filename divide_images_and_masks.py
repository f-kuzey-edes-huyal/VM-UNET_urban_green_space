import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)

# Set the path to the data folder (adjusted for Linux paths)
data_folder = Path("/mnt/c/Users/Kuzey/omdene_images_masks")

# Define source folders
image_folders = [
    data_folder / "images/SLMO_8R_1",
    data_folder / "images/SLMO_9R_2",
    data_folder / "images/SLMO_9R_3",
    data_folder / "images/VBWVA_8R"
]

mask_folders = [
    data_folder / "masks/SLMO_8R_1_masks",
    data_folder / "masks/SLMO_9R_2_masks",
    data_folder / "masks/SLMO_9R_3_masks",
    data_folder / "masks/VBWVA_8R_masks"
]

# Define destination folders
output_base = "/mnt/d/VM-UNet/data"
train_images_dir = os.path.join(output_base, "train", "images")
train_masks_dir = os.path.join(output_base, "train", "masks")
val_images_dir = os.path.join(output_base, "val", "images")
val_masks_dir = os.path.join(output_base, "val", "masks")

# Create directories if they don't exist
for directory in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
    os.makedirs(directory, exist_ok=True)

def split_and_copy_files(image_folder, mask_folder, train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, split_ratio=0.9):
    """Splits images and masks into train and validation sets and copies them to respective folders."""
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".tif", ".tiff"))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith((".png", ".jpg", ".tif", ".tiff"))])
    
    # Ensure that images and masks match
    assert len(image_files) == len(mask_files), f"Mismatch in image and mask count: {len(image_files)} images, {len(mask_files)} masks"

    # Shuffle indices
    indices = list(range(len(image_files)))
    random.shuffle(indices)
    
    # Split data into training and validation sets
    train_size = int(len(indices) * split_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Copy training data
    for idx in train_indices:
        shutil.copy(os.path.join(image_folder, image_files[idx]), os.path.join(train_images_dir, image_files[idx]))
        shutil.copy(os.path.join(mask_folder, mask_files[idx]), os.path.join(train_masks_dir, mask_files[idx]))

    # Copy validation data
    for idx in val_indices:
        shutil.copy(os.path.join(image_folder, image_files[idx]), os.path.join(val_images_dir, image_files[idx]))
        shutil.copy(os.path.join(mask_folder, mask_files[idx]), os.path.join(val_masks_dir, mask_files[idx]))

# Process all folders
for img_folder, mask_folder in zip(image_folders, mask_folders):
    split_and_copy_files(img_folder, mask_folder, train_images_dir, train_masks_dir, val_images_dir, val_masks_dir)

print("Data split completed. Check the train and val folders.")
