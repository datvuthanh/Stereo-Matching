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


def corr_layer_inference(left, right):
    return np.sum(left * right, axis=-1)


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
        "util_root": "/home/marco/repos/EfficientStereoMatching/data/KITTI2015/debug_15/",
        "data_root": "/home/marco/repos/EfficientStereoMatching/data/KITTI2015/data_scene_flow/testing",
        "experiment_root": "/home/marco/repos/EfficientStereoMatching/experiments",
        "filename": data_lookup["train"],
        "num_tr_img": 160,  # TODO: Avoid hardcoding this
        "num_val_img": 40,
        "start_id": 0,
        "num_imgs": 5
    }

    experiment_name = "020418_203327"
    experiment_path = os.path.join(args["experiment_root"], experiment_name)
    mdl = load_experiment(experiment_path)

    dh = DataHandler(args)
    dh.load()

    for i in range(args["start_id"], args["start_id"] + args["num_imgs"]):
        l_img, r_img = (dh.ldata[i][np.newaxis, ...],
                        dh.rdata[i][np.newaxis, ...])

        l_branch, r_branch = mdl.predict([l_img, r_img])
        pred = corr_layer_inference(l_branch, r_branch)
        map_width = l_branch.shape[2]
        unary_vol = np.zeros((l_branch.shape[1], map_width, 100))

        # TODO:  Implement the inference phase according to the offical
        # implementation
        plt.figure()
        plt.imshow(pred)
        plt.figure()
        plt.imshow(l_img[0])
        plt.show()

if __name__ == "__main__":
    main()
