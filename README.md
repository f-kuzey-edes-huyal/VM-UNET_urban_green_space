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
```
Pixel Counts per Class: [17208184. 14356287. 19162672.  3220142.   542660.]
Class Weights: [0.02490574 0.02985331 0.02236549 0.13309433 0.78978115]
```

__February 12, 2025__

✅ To-Do List for Code Corrections:
 - Recalculate Mean and Standard Deviation for the cleaned dataset.

```
Mean of each channel: tensor([1246.4872, 1049.9016, 1010.0692,  878.8740, 1125.1187, 2105.3875,
        2552.0520, 2452.0959, 2758.8291,  687.0012,    9.6040, 1725.2911])
Standard Deviation of each channel: tensor([ 497.1459,  553.8893,  571.6847,  676.8869,  623.9463,  923.2581,
        1142.2109, 1161.6038, 1253.5480,  350.6316,    3.7974,  859.2806])
Overall Mean of all channels: tensor(1466.7261)
Overall Standard Deviation of all channels: tensor(1139.6169)
```

 - Decrease the Number of Parameters (undo the 2x increase while maintaining model efficiency).
 - Remove Augmented Data (exclude NPY_datasets2).
 - Replace SiLU Activation with GELU and test another activation function.
 - Train for 50 Epochs with Early Stopping (Patience = 5).
 - Check If Training Stops Correctly (if not, inspect data handling).
 - Analyze and Write About the Results (performance, issues, observations).

__February 13 2025__

I made so many adjustments, but the problem still exists. I think mixed pixels might be the issue. I will try to address this problem by adjusting the mask values. I will examine the mask values, and if a pixel seems to belong to two or more classes, I will assign it to the majority class and ensure that the sum of probabilities across the 5 channels is 1 for each pixel. I am not sure if this approach will help, but I will give it a try.

__February 17 2025__

Zoran Radovanovic and Marius Hamacher shared that they have a similar issue to mine. As discussed in the project meeting, I will try to reduce the problem to a binary classification: trees versus all (possibly a low vegetation versus all problem).

```
Mean of each channel: tensor([1245.0645, 1049.0349, 1009.9710,  879.5040, 1125.7018, 2106.4487,
        2554.2410, 2454.0356, 2761.1414,  686.4368,    9.6029, 1727.0878])
Standard Deviation of each channel: tensor([ 497.6835,  554.0525,  571.9024,  677.2979,  624.4200,  925.6243,
        1145.3727, 1164.2604, 1256.9514,  352.7239,    3.8082,  860.2465])
Overall Mean of all channels: tensor(1467.3558)
Overall Standard Deviation of all channels: tensor(1141.4292)
```

__Impervious Surfaces__ Class Weights : [0.4615709129611783, 2.1665143359759824]

__Low Vegetation__ Class Weights : [0.35771197879364625, 2.795545185185919]

__Trees__ Class Weights : [0.5424329055758904, 1.8435459754017673] 

__Water__ Class Weights : [0.06280778112742051, 15.921594140874559]

__Clutter__ Class Weights : [0.010059080647556513, 99.41266354605202]

Test Results -> Mean IoU: 0.0350, Mean F1 Score: 0.0677, Mean Accuracy: 0.8220, Mean Specificity: 0.9807, Mean Sensitivity: 0.0384 (weıgted cross Ç>0.5)

Test Results -> Mean IoU: 0.5646, Mean F1 Score: 0.7217, Mean Accuracy: 0.5646, Mean Specificity: 0.0000, Mean Sensitivity: 1.0000 (argmax) 1 epoch trial

__Summary__

- I have calculated the mean and standard deviation for both the 19 individual channels of the images and the entire images.
- I checked the performance of using channel-wise mean-standard deviation versus whole-image mean-standard deviation but did not observe a significant improvement in multi-class segmentation.
- My VM-UNet classifier does not seem to be learning, and the loss stops improving after several epochs.
- I experimented with different learning rates, using 0.001, 0.01, and 0.1.
- I changed the activation function from SiLU to GELU to see if it made a difference.
- I switched the learning rate scheduler from CosineAnnealingLR to ReduceLROnPlateau.
- I used three-channel images with pre-trained VMamba weights for initialization, but this also did not make any difference.
- I calculated class weights to handle class imbalance.
- I tried different loss functions, including Cross-Entropy, Weighted Cross-Entropy, Dice Loss, Focal Loss, and a combination of Dice Loss and Cross-Entropy (both with equal weights and also with a weighting of -0.7 for Dice Loss and 0.3 for Cross-Entropy).
- I simplified the problem to one-versus-all (trees vs. all other classes).
- I recalculated class weights to address class imbalance and used them with Cross-Entropy Loss, but this did not help.
- I investigated whether I was making a mistake during image resizing, as incorrect interpolation could affect target labels. I changed the interpolation mode from InterpolationMode.BILINEAR to InterpolationMode.NEAREST, but this also did not result in a meaningful improvement.
- VM-UNet still does not seem to learn.

__Different dataset__

```
Mean of each channel: tensor([ 613.2190,  664.5815,  872.7504,  937.2291, 1248.7046, 2018.3363,
        2251.7087, 2250.5620, 2321.8311, 2339.9089, 1837.2964, 1444.3446])
Standard Deviation of each channel: tensor([245.8777, 489.7986, 482.8977, 590.6274, 459.5702, 638.2022, 791.4380,
        910.0814, 848.8351, 672.3589, 505.7607, 564.3876])
Overall Mean of all channels: tensor(1566.7062)
Overall Standard Deviation of all channels: tensor(903.7628)

```
```
Test Results -> Mean IoU: 0.7641, Mean F1 Score: 0.8663, Mean Accuracy: 0.7641, Mean Specificity: 0.0000, Mean Sensitivity: 1.0000
```
## Notes

[An app to draw architectures ](https://app.diagrams.net/)

[Cleaned data set](https://dagshub.com/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/src/main/MULC)
