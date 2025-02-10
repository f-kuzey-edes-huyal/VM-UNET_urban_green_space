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

I think there might be inconsistencies in the dataset! I initially updated the code for 12 channels but noticed that the recent images have 19 channels. I asked Dorothea about this, and she confirmed the issue. 

[] (https://stackoverflow.com/questions/71011333/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-7-768-at-en)

After many errors, I finally managed to run the code on a small subset of Omdena images! Can someone congratulate me? 😊


![](https://github.com/f-kuzey-edes-huyal/VM-UNET_urban_green_space/blob/main/running_code_for_omdena.png)


__Struggles, struggles, struggles!___ My initial segmentation attempts resulted in very poor performance:

Test Results:

- Mean IoU: 0.0905
- Mean F1 Score: 0.1623
- Mean Accuracy: 0.6834
- Mean Specificity: 0.7999
- Mean Sensitivity: 0.1995

Initially, I trained the model by assigning both loss function components (categorical cross-entropy and dice loss) equal weights of 1. However, I noticed increasing loss values, which indicated a class imbalance issue. To address this, I adjusted the weights to 0.3 for cross-entropy and 0.7 for dice loss, aiming to improve class balance handling.

The training process suggested in the referenced article is slightly different, as they use CosineAnnealingLR instead of a fixed learning rate. This scheduling method starts with a higher learning rate and gradually decreases it over time. You can read more about it [here](https://wiki.cloudfactory.com/docs/mp-wiki/scheduler/cosineannealinglr).

Since I also implemented early stopping, my training stopped at epoch 5, but I don’t think setting patience to 5 was the most reasonable choice for this type of training.

My Next Steps:

- Clean the dataset initially.
- Train using a fixed learning rate (e.g., lr = 0.01) for 50 epochs with patience set to 5.
- Once I finalize the best architecture, I will retrain using CosineAnnealingLR.

__February 6, 2025:__ The previous days were painful because my training loss started to increase after a few epochs. I experimented with different loss functions—Dice loss, cross-entropy, and a smoothed version combining both. I also adjusted the learning rate algorithm, but none of these changes helped.

Then, upon reviewing my approach, I noticed two errors. First, there was an issue with the normalization of the TIFF files—I had divided by 255 when calculating the mean, which was not suitable for them (thanks to Marius Hamacher for discussing the .tif files with me). Additionally, I decided to normalize each image channel-wise instead of using a single mean, as the paper suggested.

However, the main issue was not normalization. My biggest mistake was forgetting to binarize the mask, I believe. The algorithm is now running, and while the validation loss still oscillates, the training loss continues to decrease, which was not the case in previous experiments.



__February 7, 2025:__ 

I was expecting to get my results after coming back from shopping, but when I checked, I saw that my GPU had suddenly stopped running. I faced an issue similar to the one I mentioned earlier. I feel so disappointed because I was hoping to see some improvements in my results.

One thing I didn’t like about my training process is that the training loss started increasing again, even though the validation loss showed some stable behavior, which wasn’t the case in the initial steps.

By the way, why does a GPU stop running suddenly? Could it be due to overheating or something else?

[initialization problems ???](https://stackoverflow.com/questions/55171799/training-loss-decrease-at-first-several-epochs-but-jump-to-a-high-value-suddenly)

[class imbalance ???](https://datascience.stackexchange.com/questions/105031/why-is-my-training-loss-not-changing)

__February 10, 2025__
````
Pixel Counts per Class: [17208184. 14356287. 19162672.  3220142.   542660.]
Class Weights: [0.02490574 0.02985331 0.02236549 0.13309433 0.78978115]
```

## Notes

[An app to draw architectures ](https://app.diagrams.net/)

[Cleaned data set](https://dagshub.com/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/src/main/MULC)
