# Efficient-Deep-Learning-for-Stereo-Matching-Keras

This is a Keras version re-implementation of Luo, W., & Schwing, A. G. (n.d.). Efficient Deep Learning for Stereo Matching.
(https://www.cs.toronto.edu/~urtasun/publications/luo_etal_cvpr16.pdf)

### Setup data folders

```
kitti2015
    │─── training
         |───image_2
             |───000000_10.png
             |───000001_10.png
             |─── ...
         |───image_3
         |───disp_noc_0
         |─── ...
    │─── testing
         |───image_2
         |───image_3
```

## Preprocess data

Modify variable data_root in preprocess/kitti2015_gene_loc_1.m with the path to corresponding training folder.
Go to preprocess folder and use matlab to run: kitti2015_gene_loc_1(160,40,18,100,'debug_15',123) to generate three binary files(~300MB total), corresponding to pixel locations you want to train and validate on.

Parameters: 160 is number of images to train on, 40 is number of image to validate on, 18 represents size of image patch with (2x18+1) by (2x18+1), 100 represents searching range(disparity range to train on, corresponding to 2x100+1), 'debug_15' is the folder to save results, 123 is the random seed.

1. Matlab or Octave needs to be installed
2. Replace path to dataset in preprocess/preprocess.m
3. cd preprocess
4. octave preprocess.m


## Train

    python3 train.py --data_root /kitti2015/training --util_root preprocess/debug_15

## Test

    python3 train.py --data_root /kitti2015/training --util_root preprocess/debug_15 --phase evaluate

## Inference

    python3 inference.py --data_root /kitti2015/testing --util_root preprocess/debug_15 --model_dir checkpoint --out_dir disp_images --num_imgs 5


## Results
![Left image](predictions/000161_10.png)

Iterations: 5000, Batch size: 128, Depth: 9, Kernel size: 5 x 5

![Predicted disparity](predictions/disp_map_000161_10_5000.png)

Iterations: 20000, Batch size: 128, Depth: 9, Kernel size: 5 x 5

![Predicted disparity](predictions/disp_map_000161_10_20000.png)
