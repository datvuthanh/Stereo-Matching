from utils.data_utils.DataHandler import DataHandler
from keras.models import load_model, Model
from keras.layers import Input
from keras.backend import set_learning_phase
import keras.backend as K
import numpy as np
import os
import tensorflow as tf
from models.models import dot_win37_dep9
import matplotlib.pyplot as plt
from scipy import misc
import cv2


def map_inner_product(lmap, rmap):
	prod = tf.reduce_sum(tf.multiply(lmap, rmap), axis=3, name='map_inner_product')
	return prod


def load_experiment(experiment_path, im_shape=(375, 1242, 3)):
    path = os.path.join(experiment_path, "siamese_net.hdf5")
    mdl = dot_win37_dep9(im_shape, im_shape)
    mdl.build_model(add_corr_layer=False)
    mdl.model.load_weights(os.path.join(experiment_path, "siamese_net.hdf5"))

    return mdl.model


def main():
    set_learning_phase(0)
    data_lookup = {
        "train": "tr_160_18_100.bin",
        "val": "val_40_18_100.bin"
    }

    args = {
        "batch_size": 32,
        "data_version": "kitti2015",
        "util_root": "/content/Efficient-Deep-Learning-for-Stereo-Matching-Keras/preprocess/debug_15",
        "data_root": "/content/Efficient-Deep-Learning-for-Stereo-Matching-Keras/kitti_2015/testing",
        "experiment_root": "/content/Efficient-Deep-Learning-for-Stereo-Matching-Keras/content/Efficient-Deep-Learning-for-Stereo-Matching-Keras/experiments",
        "filename": data_lookup["val"],
        "num_tr_img": 160,  # TODO: Avoid hardcoding this
        "num_val_img": 40,
        "start_id": 0,
        "num_imgs": 5,
        "disp_range" : 201,
        "out_dir": "/content/Efficient-Deep-Learning-for-Stereo-Matching-Keras/predictions/"
    }
    
    scale_factor = 255 / (args["disp_range"] - 1)

    experiment_name = "011020_175120"
    experiment_path = os.path.join(args["experiment_root"], experiment_name)
    mdl = load_experiment(experiment_path)

    dh = DataHandler(args)
    dh.load()
    #print(dh.data_path)
    for i in range(args["start_id"], args["start_id"] + args["num_imgs"]):

        # Read image
        l_img, r_img = dh.ldata[i],dh.rdata[i]
        
        # Normalize data
        linput = (l_img - l_img.mean()) / l_img.std()
        rinput = (r_img - r_img.mean()) / r_img.std()

        # Reshape into batch 1
        linput = linput.reshape(1, linput.shape[0], linput.shape[1],linput.shape[2])
        rinput = rinput.reshape(1, rinput.shape[0], rinput.shape[1], rinput.shape[2])

        # Predict two feature 
        l_branch, r_branch = mdl.predict([linput, rinput])

        map_width = l_branch.shape[2]

        # Create cost volume with disparity range
        cost_volume = np.zeros((l_branch.shape[1], l_branch.shape[2], args["disp_range"]))
        
        for loc in range(args["disp_range"]):
            x_off = -loc
            l = l_branch[:, :, max(0, -x_off): map_width,:]
            r = r_branch[:, :, 0: min(map_width, map_width + x_off),:]
            res = map_inner_product(l, r)
            cost_volume[:, max(0, -x_off): map_width, loc] = res[0, :, :] 
        
        print('Image %s processed.' % (i + 1))

        # Get minimum cost
        pred = np.argmax(cost_volume, axis=2) * scale_factor
        
        # To do: cost aggreation --> SGM algorithm.
        
        # Get ID image
        id_img = dh.file_ids[i]
        misc.imsave('%s/disp_map_%06d_10.png' % (args["out_dir"], id_img), pred)


if __name__ == "__main__":
    main()
