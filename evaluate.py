import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization
import os
import numpy as np
from data_handler import Data_handler
from keras.utils.vis_utils import plot_model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from scipy import misc
from sgm import *
from models.model import base_model

def apply_cost_aggregation(cost_volume):
    """Apply cost-aggregation post-processing to network predictions.
    Performs an average pooling operation over raw network predictions to smoothen
    the output.
    Args:
        cost_volume (tf.Tensor): cost volume predictions from network.
    Returns:
        cost_volume (tf.Tensor): aggregated cost volume.
    """
    # NOTE: Not ideal but better than zero padding, since we average.
    cost_volume = tf.pad(cost_volume, tf.constant([[0, 0,], [2, 2,], [2, 2], [0, 0]]),
                         "REFLECT")
    # Convert float64 to float32
    cost_volume = tf.cast(cost_volume,dtype=tf.float32) 
    # Average-pooling                 
    last_layer = tf.keras.layers.AveragePooling2D(pool_size=(5, 5),
                                       strides=(1, 1),
                                       padding='VALID',
                                       data_format='channels_last')(cost_volume)
    return last_layer
                                
if __name__ == '__main__':
  flags = tf.compat.v1.app.flags

  flags.DEFINE_integer('batch_size', 128, 'Batch size.')
  flags.DEFINE_integer('num_iter', 40000, 'Total training iterations')
  flags.DEFINE_string('model_dir', 'checkpoint', 'Trained network dir')
  flags.DEFINE_string('out_dir', 'disp_images', 'output dir')
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
  flags.DEFINE_integer('num_imgs', 5, 'Number of test images')
  flags.DEFINE_integer('start_id', 0, 'ID of first test image')
  flags.DEFINE_bool('cost_aggregation', True, 'Cost aggregation')
  flags.DEFINE_bool('average_pooling', True, 'True: average pooling, False: Semi global matching')
  #flags.DEFINE_bool('post_processing',True,'Apply median blur filter after finishing cost aggreation')

  FLAGS = flags.FLAGS

  np.random.seed(123)

  #file_ids = np.fromfile(os.path.join(FLAGS.util_root, 'myPerm.bin'), '<f4')
  # Load file_ids from val.npy
  with open('val.npy', 'rb') as f:
    file_ids = np.load(f)
  #print(file_ids)

  if FLAGS.data_version == 'kitti2015':
      num_channels = 3
  elif FLAGS.data_version == 'kitti2012':
      num_channels = 1

  scale_factor = 255 / (FLAGS.disp_range - 1)

  if not os.path.exists(FLAGS.out_dir):
          os.makedirs(FLAGS.out_dir)

  # Create Finally model
  model = base_model((None,None,3))

  learning_rate = 0.01
  optimizer = optimizers.Adam(learning_rate)
  ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
  manager = tf.train.CheckpointManager(ckpt, FLAGS.model_dir, max_to_keep=3)

  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  for i in range(0, FLAGS.num_val_img):
      file_id = file_ids[i]
      if FLAGS.data_version == 'kitti2015':
          linput = misc.imread(('%s/image_2/%06d_10.png') % (FLAGS.data_root, file_id))
          rinput = misc.imread(('%s/image_3/%06d_10.png') % (FLAGS.data_root, file_id))
      
      elif FLAGS.data_version == 'kitti2012':
          linput = misc.imread(('%s/image_0/%06d_10.png') % (FLAGS.data_root, file_id))
          rinput = misc.imread(('%s/image_1/%06d_10.png') % (FLAGS.data_root, file_id))
      
      linput = (linput - linput.mean()) / linput.std()
      rinput = (rinput - rinput.mean()) / rinput.std()

      linput = linput.reshape(1, linput.shape[0], linput.shape[1], num_channels)
      rinput = rinput.reshape(1, rinput.shape[0], rinput.shape[1], num_channels)      

      # Get shape of output model
      limage_map = model(linput,training=False)
      rimage_map = model(rinput,training=False)

      # Test
      #print("Left model weights",left_model.get_weights()[0])
      #print("Right model weights",right_model.get_weights()[0])

      map_width = limage_map.shape[2]
      cost_volume = np.zeros((limage_map.shape[1], limage_map.shape[2], FLAGS.disp_range))
      #print("Cost volume shape: ",cost_volume.shape)
      for loc in range(FLAGS.disp_range):
          x_off = -loc
          l = limage_map[:, :, max(0, -x_off): map_width,:]
          r = rimage_map[:, :, 0: min(map_width, map_width + x_off),:]
          inner = tf.reduce_sum(tf.multiply(l, r), axis=3, name='map_inner_product') # Batch x 1 x 201
          #print("Inner product: ",inner.shape)
          cost_volume[:, max(0, -x_off): map_width, loc] = inner[0, :, :]
      if FLAGS.cost_aggregation:
        # We will use average pool 5x5 on original paper
        if FLAGS.average_pooling:
          print("Average pooling")
          cost_volume = cost_volume.reshape((1, cost_volume.shape[0], cost_volume.shape[1], cost_volume.shape[2]))
          cost_volume = apply_cost_aggregation(cost_volume)
          cost_volume = tf.squeeze(cost_volume)
        # If not we will use semi global matching
        else:
          print("Semi-global matching")
          parameters = Parameters(max_disparity=FLAGS.disp_range, P1=10, P2=120, csize=(7, 7), bsize=(3, 3))
          paths = Paths()
          #print("here")
          cost_volume = aggregate_costs(cost_volume, parameters, paths)
          cost_volume = np.sum(cost_volume, axis=3)
      
      pred = tf.argmax(cost_volume, axis=2)
      # Convert tensor to array
      pred = pred.numpy() * scale_factor
      # if FLAGS.post_processing:
      #   # Applying median filter size 3x3
      #   pred = cv2.medianBlur(pred,(3,3))

      misc.imsave('%s/disp_map_%06d_10.png' % (FLAGS.out_dir, file_id), pred)
      print('Image %s processed.' % (i + 1))
