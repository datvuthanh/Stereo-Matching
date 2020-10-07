import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from data_handler import Data_handler
from keras.utils.vis_utils import plot_model
from tensorflow.keras import optimizers
from models.model import base_model

def create_model(left_input,right_input):

  left_model = base_model(left_input,reuse=False)
  right_model = base_model(right_input,reuse=True) # Use weight from left_model

  # Feature extractor last layer of model
  prod = tf.reduce_sum(tf.multiply(left_model.output, right_model.output), axis=3, name='map_inner_product') # Batch x 1 x 201

  flatten = tf.keras.layers.Flatten()
  prod_flatten = flatten(prod)
  # Final model
  final_model = Model(inputs=[left_model.input, right_model.input],outputs=prod_flatten)
  return left_model,right_model,final_model

def map_inner_product(lmap, rmap):
  #prod = tf.reduce_sum(tf.multiply(lmap, rmap), axis=3, name='map_inner_product') # Batch x 1 x 201
  #print(prod)
  lbranch2 = tf.squeeze(lmap, [1])
  rbranch2 = tf.transpose(tf.squeeze(rmap, [1]), perm=[0, 2, 1])
  prod = tf.matmul(lbranch2, rbranch2)
  #print(prod)
  flatten = tf.keras.layers.Flatten()
  prod_flatten = flatten(prod)
  return prod_flatten

def call(left_input, right_input, training=None, mask=None, inference=False):
    left_feature = left_model(left_input,training=training)
    right_feature = right_model(right_input,training=training)
    inner_product = map_inner_product(left_feature,right_feature)

    return inner_product

def loss_fn(left_patches,right_patches, patch_targets,training=None):
    """Loss function."""
    inner_product = call(left_patches,right_patches,training=training)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=patch_targets, logits=inner_product), name='loss')
    return loss

def grads_fn(left_patches,right_patches, patch_targets, training=None):
    """Compute loss and gradients."""
    
    with tf.GradientTape() as tape:
        loss = loss_fn(left_patches,right_patches, patch_targets, training)
    return tape.gradient(loss, left_model.trainable_variables), loss

if __name__ == '__main__':
  flags = tf.compat.v1.app.flags

  flags.DEFINE_integer('batch_size', 128, 'Batch size.')
  flags.DEFINE_integer('num_iter', 40000, 'Total training iterations')
  flags.DEFINE_string('model_dir', 'new', 'Trained network dir')
  flags.DEFINE_string('data_version', 'kitti2015', 'kitti2012 or kitti2015')
  flags.DEFINE_string('data_root', '/content/Stereo-Matching/kitti_2015/training', 'training dataset dir')
  flags.DEFINE_string('util_root', '/content/Stereo-Matching/preprocess/debug_15', 'Binary training files dir')
  flags.DEFINE_string('net_type', 'win37_dep9', 'Network type: win37_dep9 pr win19_dep9')

  flags.DEFINE_integer('eval_size', 200, 'number of evaluation patchs per iteration')
  flags.DEFINE_integer('num_tr_img', 160, 'number of training images')
  flags.DEFINE_integer('num_val_img', 40, 'number of evaluation images')
  flags.DEFINE_integer('patch_size', 37, 'training patch size')
  flags.DEFINE_integer('num_val_loc', 10000, 'number of validation locations')
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
  
  # Create Finally model
  left_model,right_model,final_model = create_model(left_input,right_input)
  
  # Plot model
  #plot_model(final_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  #final_model.summary()


  # Create optimizer and checkpoint
  learning_rate = 0.01
  optimizer = optimizers.Adam(learning_rate)
  ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=left_model)
  manager = tf.train.CheckpointManager(ckpt, FLAGS.model_dir, max_to_keep=3)

  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  loss_history = []
  for _ in range(FLAGS.num_iter):
    lpatch, rpatch, patch_targets = dhandler.next_batch()  
    grads, t_loss = grads_fn(lpatch,rpatch,patch_targets,training=True)
    loss_history.append(t_loss.numpy().mean())
    optimizer.apply_gradients(zip(grads, left_model.trainable_variables))
    ckpt.step.assign_add(1)
    if int(ckpt.step) % 100 == 0:
      save_path = manager.save()
      print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
      print('Loss at step: %d: %.6f' % (int(ckpt.step), t_loss))
    
    if int(ckpt.step) == 24000:
      learning_rate = learning_rate / 5.
      optimizer.lr.assign(learning_rate)
    elif int(ckpt.step) > 24000 and (it - 24000) %  8000 == 0:
      learning_rate = learning_rate / 5.      
      optimizer.lr.assign(learning_rate)
  
  # plt.plot(loss_history)
  # plt.xlabel('Batch #')
  # plt.ylabel('Loss [entropy]')
