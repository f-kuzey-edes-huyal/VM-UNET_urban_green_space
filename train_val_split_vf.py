import os
import random
import shutil

# Define paths
clean_image_folder = "omdennavf/clean_cropped_images/"
clean_mask_folder = "omdennavf/clean_cropped_masks/"
train_image_folder = "omdennavf/train_images/"
train_mask_folder = "omdennavf/train_masks/"
val_image_folder = "omdennavf/val_images/"
val_mask_folder = "omdennavf/val_masks/"

# Ensure output directories exist
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_mask_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(val_mask_folder, exist_ok=True)

# Step 1: Group images by their identifiers
image_filenames = [f for f in os.listdir(clean_image_folder) if f.endswith('.tif')]
mask_filenames = [f for f in os.listdir(clean_mask_folder) if f.endswith('.tif')]

# Create a dictionary where the key is the common part of the filename (e.g., VBWVA_2016_0) and the value is a list of corresponding files
grouped_images = {}

for img in image_filenames:
    # Get the base identifier from the image filename (e.g., VBWVA_2016_0 from VBWVA_2016_0_1_GeoTIFF)
    image_base = "_".join(img.split('_')[:-1])  # Remove the last part like '_GeoTIFF'
    
    # Construct corresponding mask filename (e.g., VBWVA_2016_0_1_fractional_mask.tif from VBWVA_2016_0_1_GeoTIFF)
    mask_base = img.replace(".tif", "_fractional_mask.tif")
    
    if mask_base not in mask_filenames:
        print(f"Warning: Mask not found for {img}. Skipping.")
        continue
    
    # Add image and mask to grouped data
    if image_base not in grouped_images:
        grouped_images[image_base] = {'images': [], 'masks': []}
    
    grouped_images[image_base]['images'].append(img)
    grouped_images[image_base]['masks'].append(mask_base)

# Step 2: Split the groups into training and validation sets
train_groups = random.sample(list(grouped_images.keys()), int(0.8 * len(grouped_images)))
val_groups = [group for group in grouped_images if group not in train_groups]

# Step 3: Copy the files into the respective directories
def copy_files(grouped_files, image_folder, mask_folder, target_image_folder, target_mask_folder):
    for group in grouped_files:
        for img, mask in zip(grouped_files[group]['images'], grouped_files[group]['masks']):
            shutil.copy(os.path.join(image_folder, img), os.path.join(target_image_folder, img))
            shutil.copy(os.path.join(mask_folder, mask), os.path.join(target_mask_folder, mask))

# Copy training and validation images and masks
copy_files({group: grouped_images[group] for group in train_groups}, clean_image_folder, clean_mask_folder, train_image_folder, train_mask_folder)
copy_files({group: grouped_images[group] for group in val_groups}, clean_image_folder, clean_mask_folder, val_image_folder, val_mask_folder)

print("Data splitting completed successfully!")
