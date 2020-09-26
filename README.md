# Efficient-Deep-Learning-for-Stereo-Matching-Keras

This is a Keras version re-implementation of Luo, W., & Schwing, A. G. (n.d.). Efficient Deep Learning for Stereo Matching.
(https://www.cs.toronto.edu/~urtasun/publications/luo_etal_cvpr16.pdf)


## To run

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
1. Matlab or Octave needs to be installed
2. Replace path to dataset in preprocess/preprocess.m
3. cd preprocess
4. octave preprocess.m


## Train

    python3 train.py --data /kitti2015/training

## Test

    python3 eval.py --data /kitti2015/training --checkpoint pretrained pretrained/win37_dep9.pkl

## Inference

    python3 inference.py --data /kitti2015/testing --img_num 0


## Results
Iterations: 40000, Batch size: 128, Depth: 9, Kernel size: 5 x 5

![Left image](images/win37_dep9_left.png)
![Predicted disparity](images/win37_dep9.png)
