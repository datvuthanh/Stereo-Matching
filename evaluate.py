import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from data_handler import Data_handler
from keras.utils.vis_utils import plot_model
from tensorflow.keras import optimizers
from models.model import base_model
from tensorflow.keras.models import Model

def map_inner_product(lmap, rmap):
  lbranch2 = tf.squeeze(lmap, [1])
  rbranch2 = tf.transpose(tf.squeeze(rmap, [1]), perm=[0, 2, 1])
  prod = tf.matmul(lbranch2, rbranch2)
  flatten = tf.keras.layers.Flatten()
  prod_flatten = flatten(prod)
  return prod_flatten


if __name__ == '__main__':
  flags = tf.compat.v1.app.flags

  flags.DEFINE_integer('batch_size', 128, 'Batch size.')
  flags.DEFINE_integer('num_iter', 100000, 'Total training iterations')
  flags.DEFINE_string('model_dir', 'checkpoint', 'Trained network dir')
  flags.DEFINE_string('data_version', 'kitti2015', 'kitti2012 or kitti2015')
  flags.DEFINE_string('data_root', './kitti2015/training', 'training dataset dir')
  flags.DEFINE_string('util_root', './preprocess/debug_15', 'Binary training files dir')
  flags.DEFINE_string('net_type', 'win37_dep9', 'Network type: win37_dep9 pr win19_dep9')

  flags.DEFINE_integer('eval_size', 200, 'number of evaluation patchs per iteration')
  flags.DEFINE_integer('num_tr_img', 160, 'number of training images')
  flags.DEFINE_integer('num_val_img', 40, 'number of evaluation images')
  flags.DEFINE_integer('patch_size', 37, 'training patch size')
  flags.DEFINE_integer('num_val_loc', 1000, 'number of validation locations')
  flags.DEFINE_integer('disp_range', 201, 'disparity range')
  flags.DEFINE_string('phase', 'train', 'train or evaluate')


  FLAGS = flags.FLAGS

  np.random.seed(123)

  # Load Dataset
  dhandler = Data_handler(data_version=FLAGS.data_version, 
    data_root=FLAGS.data_root,  
    util_root=FLAGS.util_root, 
    num_tr_img=FLAGS.num_tr_img, 
    num_val_img=FLAGS.num_val_img, 
    num_val_loc=FLAGS.num_val_loc, 
    batch_size=FLAGS.batch_size, 
    patch_size=FLAGS.patch_size, 
    disp_range=FLAGS.disp_range)  
  
  # Create left model, right model
  if FLAGS.data_version == 'kitti2015':
      num_channels = 3
  elif FLAGS.data_version == 'kitti2012':
      num_channels = 1

  left_input = (FLAGS.patch_size,FLAGS.patch_size,num_channels)
  right_input = (FLAGS.patch_size,FLAGS.patch_size + FLAGS.disp_range - 1, num_channels)
  
  # Create model  
  model = base_model((None,None,3))

  # Create optimizer and checkpoint
  learning_rate = 0.001
  optimizer = optimizers.Adam(learning_rate)
  ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
  manager = tf.train.CheckpointManager(ckpt, FLAGS.model_dir, max_to_keep=3)

  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  lpatch, rpatch, patch_targets = dhandler.evaluate() 
  labels = np.argmax(patch_targets, axis = 1)

  left_feature = model(lpatch,training=False)
  right_feature = model(rpatch,training=False)
  
  # Inner product
  prod = map_inner_product(left_feature,right_feature)

  predicted = tf.argmax(prod,axis = 1) # Get best disparity range of each pixels

  acc_count = 0

  for i in range(0,lpatch.shape[0], FLAGS.eval_size):

    acc_count += np.sum(np.abs(predicted[i: i + FLAGS.eval_size] - labels[i: i + FLAGS.eval_size]) <= 3)
    
    print('iter. %d finished, with %d correct (3-pixel error)' % (i + 1, acc_count))
  
  print('accuracy: %.3f' % ((acc_count / lpatch.shape[0]) * 100))


    
