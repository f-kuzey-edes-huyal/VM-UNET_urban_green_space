import os
import numpy as np
import torch
import rasterio  # For reading GeoTIFF images
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def load_masks_from_folder(folder_path, file_extension=".tif"):
    """
    Loads all masks from a folder and stacks them into a tensor.
    
    Args:
    - folder_path (str): Path to the folder containing mask images.
    - file_extension (str): Image format (.tif, .png, .npy, etc.)
    
    Returns:
    - mask_list (list of torch.Tensors): List of loaded masks.
    """
    mask_list = []
    for filename in sorted(os.listdir(folder_path)):  # Sorted for consistency
        if filename.endswith(file_extension):
            file_path = os.path.join(folder_path, filename)

            if file_extension == ".tif":  # GeoTIFF format
                with rasterio.open(file_path) as src:
                    mask = src.read()  # Shape: (num_classes, H, W)

            elif file_extension == ".npy":  # NumPy format
                mask = np.load(file_path)  # Shape: (num_classes, H, W)

            else:  # Assume standard image format (PNG, JPG)
                mask = Image.open(file_path)
                mask = transforms.ToTensor()(mask)  # Converts to (C, H, W)
            
            mask_list.append(torch.tensor(mask, dtype=torch.float32))
    
    return mask_list

def compute_class_weights(mask_list):
    """
    Computes class weights for multi-class segmentation based on all images.
    
    Args:
    - mask_list (list of torch.Tensors): List of tensors of shape (num_classes, H, W)
    
    Returns:
    - class_weights (list): Normalized class weights.
    - pixel_counts (list): Number of pixels per class.
    """
    num_classes = mask_list[0].shape[0]  # Assume all masks have the same number of classes
    total_pixels_per_class = torch.zeros(num_classes)

    # Process each mask
    for mask in mask_list:
        size = mask.shape[1] * mask.shape[2]  # H * W
        mask = mask.unsqueeze(0)  # Add batch dimension for processing

        # Step 1: Binarize the mask
        _, max_indices = torch.max(mask, dim=1, keepdim=True)  # Get index of max probability
        binarized_mask = (torch.arange(num_classes).view(1, -1, 1, 1) == max_indices).float()

        # Step 2: Count pixels per class
        total_pixels_per_class += binarized_mask.sum(dim=(0, 2, 3))  # Sum across images

    # Step 3: Compute class weights using inverse frequency
    total_pixels = total_pixels_per_class.sum()
    class_weights = total_pixels / (total_pixels_per_class + 1e-6)  # Avoid division by zero

    # Normalize weights to sum to 1
    class_weights = class_weights / class_weights.sum()

    return class_weights.numpy(), total_pixels_per_class.numpy()

def plot_class_weights(class_weights):
    """
    Plots a bar chart of class weights.
    
    Args:
    - class_weights (list): Class weights computed for balancing.
    """
    num_classes = len(class_weights)
    plt.figure(figsize=(8, 5))
    #plt.bar(range(num_classes), class_weights, tick_label=[f'Class {i}' for i in range(num_classes)])
    plt.bar(range(num_classes), class_weights, tick_label=['Imp. surfaces', 'Low vegetation', 'Trees', 'Water', 'clutter/Backg.'], color = 'cornflowerblue')
    plt.xlabel("Class")
    plt.ylabel("Weight")
    plt.title("Class Weights for Balanced Learning")
    plt.show()

# Example Usage
folder_path = "./data/omdena/cleaned/train/masks"  # Update with actual folder path
mask_list = load_masks_from_folder(folder_path, file_extension=".tif")  # Change format if needed

class_weights, pixel_counts = compute_class_weights(mask_list)
plot_class_weights(class_weights)

# Print results
print(f"Pixel Counts per Class: {pixel_counts}")
print(f"Class Weights: {class_weights}")
