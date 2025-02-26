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

Images and masks for index 0 saved successfully as both TIFF and PNG.
100%|█████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  4.68it/s]Confusion Matrix:
 [[ 2884  3772]
 [12100 14012]]
Test Results -> Mean IoU: 0.4689, Mean F1 Score: 0.6384, Mean Accuracy: 0.5156, Mean Specificity: 0.4333, Mean Sensitivity: 0.5366

 [[ 2992  3664]
 [12658 13454]]
Test Results -> Mean IoU: 0.4518, Mean F1 Score: 0.6224, Mean Accuracy: 0.5019, Mean Specificity: 0.4495, Mean Sensitivity: 0.5152

__February 21 2025__

I found a dataset on [Zenodo](https://zenodo.org/records/8413116) that contained a single .tif file for images and another for masks. I divided the images into 16 × 16 patches and ran the same experiment, but the results were identical across different  number of epoch runs—the model was not learning. Specifically, I kept getting only white regions, with Mean Specificity: 0.0000. It was the same with my previous experiments.
Here are the results from that experiment:
Test Results:
- Mean IoU: 0.7641
- Mean F1 Score: 0.8663
- Mean Accuracy: 0.7641
- Mean Specificity: 0.0000
- Mean Sensitivity: 1.0000
  
Then, I remembered some [discussions](https://discuss.pytorch.org/t/semantic-segmentation-model-unet-doesnt-learn/52899/2) about the importance of resizing images carefully, as it introduces new values to the targets. Instead of creating 16 × 16 patches, I adjusted the process to create 128 × 128 patches—matching the input size expected by my model. I used six training images and two validation images, and __this time, I noticed that the model was finally learning__. At least, there was some variation between runs.
Here are the updated results:

__After 100 epochs:__
= Confusion Matrix: [[ 2884 3772] [12100 14012]]
- Mean IoU: 0.4689
- Mean F1 Score: 0.6384
- Mean Accuracy: 0.5156
- Mean Specificity: 0.4333
- Mean Sensitivity: 0.5366

__After 300 epochs:__
Confusion Matrix: [[ 2992 3664] [12658 13454]]
- Mean IoU: 0.4518
- Mean F1 Score: 0.6224
- Mean Accuracy: 0.5019
- Mean Specificity: 0.4495
- Mean Sensitivity: 0.5152

While the model still has room for improvement, at least it is showing some learning behavior. Given these findings, I was wondering if there might be an issue with the dataset itself.

__February 25 2025__

The cleaning process involves several key steps to ensure the satellite images and their corresponding masks are valid and contain meaningful data. First, the images and masks are validated by checking their shape, ensuring they have the correct number of channels (19 for images and 5 for masks). We also ensure that no NaN values are present, and discard any images or masks that are completely black or lack sufficient content, determined by a minimum content threshold (set at 1% pixel intensity). Additionally, we check for excessive noise or corruption by ensuring that the images and masks do not contain anomalies like missing data or pixels that are all zeros. Only images and masks that pass these checks are saved to a new folder, ensuring a cleaner dataset for further processing and model training.

To separate the training and validation sets, we first grouped the images and their corresponding masks by their common identifier (e.g., "VBWVA_2016_0"). This grouping ensures that images taken from the same area but at different time points (such as "VBWVA_2016_0_1_GeoTIFF", "VBWVA_2016_0_2_GeoTIFF", and "VBWVA_2016_0_3_GeoTIFF") are treated as a single unit, preserving the temporal correlation between the images.

Next, we randomly selected 80% of the groups to form the training set, ensuring that the images within each group (representing the same area) are kept together in the training set. The remaining 20% of the groups were designated for the validation set. This method helps maintain the temporal relationship between images in each group while dividing the data into training and validation sets.

```
Mean per channel: [119.68150329589844, 117.33367919921875, 127.63949584960938, 129.77781677246094, 124.62456512451172, 125.06695556640625, 124.95356750488281, 125.3196792602539, 124.92810821533203, 122.48779296875, 10.059982299804688, 124.89237976074219, 125.14395904541016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Std per channel: [71.22357940673828, 72.63861083984375, 78.41352081298828, 72.97818756103516, 76.74971008300781, 75.35397338867188, 75.2880859375, 75.13275909423828, 75.24949645996094, 76.35897827148438, 2.174912452697754, 75.01387023925781, 76.76856231689453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Overall Mean: 79.0478744506836
Overall Std: 84.19537353515625
```



```

Trees versus all

100%|█████████████████████████████████████████████████████████████████████████████████| 194/194 [00:12<00:00, 15.80it/s]Confusion Matrix:
 [[  56340  159758]
 [ 738284 2224114]]
Test Results -> Mean IoU: 0.7124, Mean F1 Score: 0.8320, Mean Accuracy: 0.7175, Mean Specificity: 0.2607, Mean Sensitivity: 0.7508

Early stop at epoch 29 (patience 2)


Mean per channel: [608.6400756835938, 660.2857055664062, 869.1080932617188, 932.1439208984375, 1245.96875, 2024.5614013671875, 2260.92822265625, 2260.4033203125, 2332.1083984375, 2346.77783203125, 1836.15625, 1436.677734375]
Std per channel: [250.7348175048828, 500.5494384765625, 493.658447265625, 599.875244140625, 476.97625732421875, 656.5435791015625, 809.8717651367188, 924.457763671875, 868.8622436523438, 686.1751708984375, 533.7572021484375, 580.5165405273438]
Overall Mean: 1567.8133544921875
Overall Std: 918.4216918945312

64 times 64
```Mean per channel: [628.7136840820312, 681.3084716796875, 885.525634765625, 959.0494995117188, 1255.718994140625, 1968.361083984375, 2183.942626953125, 2178.94921875, 2245.24072265625, 2265.470947265625, 1820.593505859375, 1453.2490234375]
Std per channel: [255.5467529296875, 504.3299560546875, 499.6480712890625, 602.7828979492188, 477.06463623046875, 652.8716430664062, 800.3475341796875, 915.36865234375, 857.1869506835938, 681.0606079101562, 529.3170776367188, 577.427490234375]
Overall Mean: 1543.8436279296875
Overall Std: 885.42529296875
```


__February 26 2034__
36 training images with 64*64 height*width.

__Initial three channel with pre-trained VMamba weights__


0 vs all



```Confusion Matrix:
 [[ 3828  5772]
 [10844 16420]]
Test Results -> Mean IoU: 0.4970, Mean F1 Score: 0.6640, Mean Accuracy: 0.5493, Mean Specificity: 0.3987, Mean Sensitivity: 0.6023

12 channel

Confusion Matrix:
 [[ 4117  5483]
 [11925 15339]]
Test Results -> Mean IoU: 0.4684, Mean F1 Score: 0.6380, Mean Accuracy: 0.5278, Mean Specificity: 0.4289, Mean Sensitivity: 0.5626

1 vs all

Confusion Matrix:
 [[ 3935  4833]
 [12209 15887]]
Test Results -> Mean IoU: 0.4825, Mean F1 Score: 0.6509, Mean Accuracy: 0.5377, Mean Specificity: 0.4488, Mean Sensitivity: 0.5655

12 channel

Confusion Matrix:
 [[ 3524  5244]
 [12201 15895]]
Test Results -> Mean IoU: 0.4768, Mean F1 Score: 0.6457, Mean Accuracy: 0.5268, Mean Specificity: 0.4019, Mean Sensitivity: 0.5657

2 vs all

Confusion Matrix:
 [[ 3587  5117]
 [11987 16173]]
Test Results -> Mean IoU: 0.4860, Mean F1 Score: 0.6541, Mean Accuracy: 0.5360, Mean Specificity: 0.4121, Mean Sensitivity: 0.5743

12 channel used

```Confusion Matrix:
 [[ 3542  5162]
 [11921 16239]]
Test Results -> Mean IoU: 0.4873, Mean F1 Score: 0.6553, Mean Accuracy: 0.5366, Mean Specificity: 0.4069, Mean Sensitivity: 0.5767
```


3 vs all

3 chaanel used
]Confusion Matrix:
 [[ 4069  5467]
 [12067 15261]]
Test Results -> Mean IoU: 0.4653, Mean F1 Score: 0.6351, Mean Accuracy: 0.5244, Mean Specificity: 0.4267, Mean Sensitivity: 0.5584
```

12 channel used
```Confusion Matrix:
 [[ 3966  5570]
 [11895 15433]]
Test Results -> Mean IoU: 0.4691, Mean F1 Score: 0.6386, Mean Accuracy: 0.5262, Mean Specificity: 0.4159, Mean Sensitivity: 0.5647
```

I had a meeting with Aany Sofia related to applying [PCA](https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118/) for reducing channel size to use with pretrained weights. I will try it. I love talking with her! ❤️❤️❤️
## Notes

[An app to draw architectures ](https://app.diagrams.net/)

[Cleaned data set](https://dagshub.com/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/src/main/MULC)
