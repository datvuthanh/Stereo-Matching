# Processing data

You need to install octave

``` apt install -y octave ```

Edit path on preprocess.m with your dataset path. 

```cd preprocess && octave preprocess.m```

Modify variable data_root in preprocess/kitti2015_gene_loc_1.m with the path to corresponding training folder.
Go to preprocess folder and use matlab to run: kitti2015_gene_loc_1(160,40,18,100,'debug_15',123) to generate three binary files(~300MB total), corresponding to pixel locations you want to train and validate on.

Parameters: 160 is number of images to train on, 40 is number of image to validate on, 18 represents size of image patch with (2x18+1) by (2x18+1), 100 represents searching range(disparity range to train on, corresponding to 2x100+1), 'debug_15' is the folder to save results, 123 is the random seed.

You can download my dataset here:
Edit preprocess.m if you want to create patches 9x9 
```kitti2015_gene_loc_1(160,40,4,100,'debug_15_ws_9',123)```
