import argparse
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from data_handler import DataHandler
from model import build_model

epochs = 10
learning_rate = 1e-3
batch_size = 64
checkpoint_path = './checkpoints_best_acc.h5'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root')
args = parser.parse_args()


data_loader = DataHandler(
    batch_size=batch_size,
    data_version='kitti2015',
    util_root='./preprocess/debug_15/',
    data_root=args.data_root,
    filename='tr_40_18_100.bin',
)
data_loader.load()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

checkpoints = ModelCheckpoint(checkpoint_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
model = build_model(left_input_shape=data_loader.l_psz, right_input_shape=data_loader.r_psz)
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
train_samples = data_loader.pixel_loc.shape[0]
model.fit_generator(
    generator=data_loader.generator,
    steps_per_epoch=(train_samples // batch_size) // 10,
    callbacks=[checkpoints],
    epochs=10,
)
