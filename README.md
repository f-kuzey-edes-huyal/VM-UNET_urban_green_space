# VM-UNET_urban_green_space
This repository shares my progress, code, and challenges during the VM-UNet implementation for Frankfurt green space mapping.

# To-Do List and Progress

## To-Do List:

| Task                 | Description                                                                                  |
|----------------------|----------------------------------------------------------------------------------------------|
| **Environment Setup**| Research how to construct the VM-UNet environment using Kaggle notebooks.                   |
| **Data Preparation** | Download the dataset and crop images to 128x128 during preprocessing.                        |
| **Model Modification**| Modify VM-UNet to handle GeoTIFF images for training.                                       |
| **Statistical Analysis**| Calculate the mean and standard deviation of the images.                                  |
| **Code Updates**     | Modify the existing code to meet project-specific requirements.                              |
| **Class Imbalance**  | Research methods to address class imbalance in segmentation (e.g., mask regions).            |
| **Data Augmentation**| Evaluate fixed augmentations like random flip, translate, or affine transformations. Avoid rotations. |
| **Early Stopping**   | Add early stopping functionality to the original code.                                       |

## Progress:

| Task                 | Progress                                                                                     |
|----------------------|----------------------------------------------------------------------------------------------|
| **Image Augmentations**| Research and discussions completed; feedback from Isabelle and Dorothea included.???       |


```
RuntimeError: Error(s) in loading state_dict for VSSM:
        size mismatch for patch_embed.proj.weight: copying a param with shape torch.Size([96, 3, 4, 4]) from checkpoint, the shape in current model is torch.Size([96, 12, 4, 4]).
```


I think the problem is caused by the channel size of the pre-trained weights. GeoTIFF images contain 12 channels, while the pre-trained weights are designed for 3 channels. Before identifying the root cause of the issue, I made several changes to dataset.py, vmamba.py, vmunet.py, and configs.py. Unfortunately, these changes did not resolve the problem.

Now, I aim to try two approaches:

- Remove the pre-trained weights and start again.
- Reshape the pre-trained weights to match the 12-channel input and try again.
  
I hope solving this issue won’t take too much effort.

__Error__ I tried discarding the initialization with pretrained weights, but I encountered the error shown below.

 ```
File "/mnt/d/VM-UNet/models/vmunet/vmamba.py", line 468, in forward
    y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
RuntimeError: shape '[1, 64, 64, -1]' is invalid for input of size 762048
 ```

__Error__ 

```
 File "/mnt/d/VM-UNet/datasets/dataset_omdena.py", line 54, in __getitem__
    raise ValueError(f"Expected image with 12 channels, but got {img.shape[0]} channels: {img_path}")
ValueError: Expected image with 12 channels, but got 19 channels: ./data/omdena/train/images/VBWVA_2016_208_2_GeoTIFF.tif
```
__Error__

```
RuntimeError: stack expects each tensor to be equal size, but got [132, 128, 128] at entry 0 and [133, 128, 128] at entry 1
```

## Notes

[An app to draw architectures ](https://app.diagrams.net/)
