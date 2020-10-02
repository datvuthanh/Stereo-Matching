from utils.data_utils.DataHandler import DataHandler
from models.models import dot_win37_dep9, dot_win19_dep9
from keras.callbacks import (
    EarlyStopping, LearningRateScheduler, TensorBoard, ModelCheckpoint,
    ReduceLROnPlateau)
from keras.optimizers import Adagrad
import keras.backend as K
import numpy as np
import os
import tensorflow as tf
import datetime


def create_experiment(path):
    now = datetime.datetime.now()
    experiment_name = now.strftime("%d%m%y_%H%M%S")
    path = os.path.join(path, experiment_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    return {"path": path}


def get_callbacks(experiment):
    def schedule(epoch, lr):
        if epoch > 10:
            out = lr * 0.2
        elif epoch % 5 == 0 and epoch > 10:
            out = lr * 0.2
        return out

    monitor, patience, verbose = "val_loss", 10, True
    es = EarlyStopping(monitor=monitor, patience=10, verbose=verbose)
    mdlchkpt = ModelCheckpoint(
        filepath=os.path.join(experiment["path"], "siamese_net.hdf5"),
        monitor=monitor, save_best_only=True)
    tb = TensorBoard(log_dir=experiment["path"])
    lrs = ReduceLROnPlateau(patience=10, verbose=True)

    return [mdlchkpt, tb, lrs]


def main():
    data_lookup = {
        "train": "tr_160_18_100.bin",
        "val": "val_40_18_100.bin"
    }

    args = {
        "batch_size": 128,
        "data_version": "kitti2015",
        "util_root": "/content/Stereo-Matching/preprocess/debug_15",
        "data_root": "/content/Stereo-Matching/kitti_2015/training",
        "experiment_root": "/content/Stereo-Matching/experiments",
        "num_val_loc": 10000,
        "num_tr_img": 160,  # TODO: Avoid hardcoding this
        "num_val_img": 40,
    }

    dh = DataHandler(args)
    dh.load(data_lookup["train"])
    train_gen = dh.generator
    train_samples = dh.pixel_loc.shape[0]

    dh_val = DataHandler(args)
    dh_val.load(data_lookup["val"])
    val_gen = dh_val.generator
    val_samples = dh_val.pixel_loc.shape[0]

    network = dot_win37_dep9(dh.args["l_psz"], dh.args["r_psz"])
    network.build_model()
    mdl = network.model
    
    # Load weights
    mdl.load_weights('/content/Stereo-Matching/siamese_net.hdf5')

    experiment = create_experiment(args["experiment_root"])
    cbs = get_callbacks(experiment)
    optim = Adagrad()

    mdl.compile(optimizer=optim, loss="categorical_crossentropy")
    mdl.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_samples // args["batch_size"],
        epochs=10,
        verbose=1,
        callbacks=cbs,
        validation_data=val_gen,
        validation_steps=val_samples // args["batch_size"])


if __name__ == "__main__":
    main()
