from torch.utils.data import Dataset
import numpy as np
import os
import random
import torch
from scipy.ndimage import zoom
import rasterio

class NPY_datasets(Dataset):
    def __init__(self, path_data, config, train=True):
        """
        A dataset class for loading TIFF images and masks using rasterio.

        Args:
            path_data (str): The root path to the dataset containing images and masks.
            config (object): Config object containing transformers for training and testing.
            train (bool): Whether to load training or validation data.
        """
        super(NPY_datasets, self).__init__()
        folder = 'train' if train else 'val'
        images_path = os.path.join(path_data, folder, 'images')
        masks_path = os.path.join(path_data, folder, 'masks')

        # Filter only .tif files
        images_list = sorted([f for f in os.listdir(images_path) if f.endswith('.tif')])
        masks_list = sorted([f for f in os.listdir(masks_path) if f.endswith('.tif')])

        self.data = [
            (os.path.join(images_path, img), os.path.join(masks_path, msk))
            for img, msk in zip(images_list, masks_list)
        ]
        self.transformer = config.train_transformer if train else config.test_transformer

    def __getitem__(self, indx):
        """
        Get a single item from the dataset.

        Args:
            indx (int): Index of the data to fetch.
        
        Returns:
            tuple: Transformed image and mask tensors.
        """
        img_path, msk_path = self.data[indx]

        # Load image and mask using rasterio
        with rasterio.open(img_path) as img_src:
            img = img_src.read()  # Shape: (channels, height, width)
            img = img[:12, :, :]  # Select only the first 12 channels

        with rasterio.open(msk_path) as msk_src:
            msk = msk_src.read()

        # Ensure correct shape (12 channels for image, 5 for mask)
        if img.shape[0] != 12:
            raise ValueError(f"Expected image with 12 channels, but got {img.shape[0]} channels: {img_path}")
        if msk.shape[0] != 5:
            raise ValueError(f"Expected mask with 5 channels, but got {msk.shape[0]} channels: {msk_path}")

        # Apply transformations
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        """
        Get the total number of data points.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k, axes=(1, 2))
    axis = np.random.randint(1, 3)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = zoom(image, angle, order=0, reshape=False)
    label = zoom(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        """
        Random data augmentation generator.

        Args:
            output_size (tuple): Desired output dimensions (H, W).
        """
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # Resize to target output size
        if image.shape[1:] != self.output_size:
            zoom_factors = (
                1,  # No zoom on channels
                self.output_size[0] / image.shape[1],
                self.output_size[1] / image.shape[2],
            )
            image = zoom(image, zoom_factors, order=3)
            label = zoom(label, zoom_factors, order=0)

        # Convert to tensors
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        return image, label
