import os
import rasterio
import numpy as np
from PIL import Image
import torch
from pathlib import Path

def crop_center(image, size=(128, 128)):
    """Crop the center of an image to the target size."""
    w, h = image.size
    new_w, new_h = size
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = (w + new_w) // 2
    bottom = (h + new_h) // 2
    return image.crop((left, top, right, bottom))

def resize_image(image, size=(128, 128)):
    """Resize image to the target size."""
    return image.resize(size, Image.Resampling.LANCZOS)

def process_geotiff_folder(folder_path, target_size=(128, 128), channels_to_use=12):
    """Process all GeoTIFF files in the folder and return a tensor of processed images."""
    all_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(folder_path, filename)

            # Open the GeoTIFF file
            with rasterio.open(file_path) as src:
                # Read all channels (bands) into a numpy array
                image = src.read()

                # Select only the first 'channels_to_use' channels (e.g., first 12)
                image = image[:channels_to_use, :, :]  # Shape: [C, H, W]

                # Convert to float32 to avoid issues with uint32
                image = image.astype(np.float32)  # Ensure dtype is float32

                # Create a tensor by stacking the channels along the second dimension (channel axis)
                image_tensor = torch.tensor(image)  # Shape: [C, H, W]

                # Normalize the image to [0, 1]
                image_tensor = image_tensor / 255.0  # Normalize to range [0, 1]

                # Resize and crop each channel
                all_resized_channels = []
                for c in range(channels_to_use):
                    channel_image = Image.fromarray(image[c])  # Convert channel to PIL image
                    cropped_image = crop_center(channel_image, size=target_size)
                    resized_image = resize_image(cropped_image, size=target_size)
                    resized_tensor = torch.tensor(np.array(resized_image)).float() / 255.0
                    all_resized_channels.append(resized_tensor)

                # Stack all channels along the first dimension (channel axis)
                image_tensor = torch.stack(all_resized_channels, dim=0)  # Shape: [C, H, W]

                # Add to the list of processed images
                all_images.append(image_tensor)

    # Stack all images into a single tensor (Shape: [N, C, H, W])
    images_tensor = torch.stack(all_images, dim=0)  # Shape: [N, C, H, W]

    return images_tensor

def calculate_mean_std_of_channels(images_tensor, channels_to_use=12):
    """Calculate the mean and standard deviation for each of the 12 channels individually,
       and overall mean and std for the entire image (across all channels)."""
    
    # Ensure the tensor is [N, C, H, W]
    if len(images_tensor.shape) != 4:
        raise ValueError("Input tensor should have 4 dimensions: [N, C, H, W]")

    # Initialize lists to hold means and stds for each channel
    mean_channels = []
    std_channels = []
    
    # Loop over each channel
    for c in range(channels_to_use):
        channel_data = images_tensor[:, c, :, :]  # Shape: [N, H, W]
        
        # Mean and Std for each channel (across N, H, W)
        mean = channel_data.mean()
        std = channel_data.std()
        
        mean_channels.append(mean)
        std_channels.append(std)
    
    # Convert lists to tensors
    mean_channels = torch.tensor(mean_channels)
    std_channels = torch.tensor(std_channels)
    
    # Calculate overall mean and std for all channels (across N, C, H, W)
    # Flatten the tensor across all dimensions (N, C, H, W) and calculate mean and std for all pixels
    flattened_tensor = images_tensor.view(-1)  # Flatten to a 1D tensor
    
    mean_all_channels = flattened_tensor.mean()  # Single mean across all pixels
    std_all_channels = flattened_tensor.std()    # Single std across all pixels

    return mean_channels, std_channels, mean_all_channels, std_all_channels


def process_multiple_folders(folders, target_size=(128, 128), channels_to_use=12):
    """Process GeoTIFF images from multiple folders and return combined tensor of all images."""
    all_images = []
    for folder in folders:
        images_tensor = process_geotiff_folder(folder, target_size, channels_to_use)
        all_images.append(images_tensor)

    # Concatenate images from all folders into one tensor
    combined_images_tensor = torch.cat(all_images, dim=0)  # Shape: [N, C, H, W]
    return combined_images_tensor

# Set the path to the data folder (adjusted for Linux paths)
data_folder = Path("/mnt/c/Users/Kuzey/omdene_images_masks/images")

# Example usage
folders = [
    data_folder / "SLMO_8R_1",
    data_folder / "SLMO_9R_2",
    data_folder / "SLMO_9R_3",
    data_folder / "VBWVA_8R"
]

processed_images = process_multiple_folders(folders, target_size=(128, 128), channels_to_use=12)

# Calculate mean and std for each channel and overall
mean_channels, std_channels, mean_all_channels, std_all_channels = calculate_mean_std_of_channels(processed_images, channels_to_use=12)

print("Mean of each channel:", mean_channels)
print("Standard Deviation of each channel:", std_channels)
print("Overall Mean of all channels:", mean_all_channels)
print("Overall Standard Deviation of all channels:", std_all_channels)
