#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import models.net_factory as nf
import numpy as np
from data_handler import Data_handler
from scipy import misc
import matplotlib.pyplot as plt
from sgm import *

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('num_iter', 40000, 'Total training iterations')
flags.DEFINE_string('model_dir', 'checkpoint', 'Trained network dir')
flags.DEFINE_string('out_dir', 'disp_images', 'output dir')
flags.DEFINE_string('data_version', 'kitti2015', 'kitti2012 or kitti2015')
flags.DEFINE_string('data_root', '/content/Stereo-Matching/keras/kitti_2015/training', 'training dataset dir')
flags.DEFINE_string('util_root', '/content/Stereo-Matching/keras/preprocess/debug_15', 'Binary training files dir')
flags.DEFINE_string('net_type', 'win37_dep9', 'Network type: win37_dep9 pr win19_dep9')

flags.DEFINE_integer('eval_size', 200, 'number of evaluation patchs per iteration')
flags.DEFINE_integer('num_tr_img', 160, 'number of training images')
flags.DEFINE_integer('num_val_img', 40, 'number of evaluation images')
flags.DEFINE_integer('patch_size', 37, 'training patch size')
flags.DEFINE_integer('num_val_loc', 50000, 'number of validation locations')
flags.DEFINE_integer('disp_range', 201, 'disparity range')
flags.DEFINE_integer('num_imgs', 5, 'Number of test images')
flags.DEFINE_integer('start_id', 0, 'ID of first test image')
flags.DEFINE_bool('cost_aggregation', True, 'Cost aggregation')
flags.DEFINE_bool('average_pooling', False, 'True: average pooling, False: Semi global matching')
#flags.DEFINE_bool('post_processing',True,'Apply median blur filter after finishing cost aggreation')

FLAGS = flags.FLAGS

np.random.seed(123)

file_ids = np.fromfile(os.path.join(FLAGS.util_root, 'myPerm.bin'), '<f4')

if FLAGS.data_version == 'kitti2015':
    num_channels = 3
elif FLAGS.data_version == 'kitti2012':
    num_channels = 1

scale_factor = 255 / (FLAGS.disp_range - 1)

if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)


with tf.Session() as session:

        limage = tf.placeholder(tf.float32, [None, None, None, num_channels], name='limage')
        rimage = tf.placeholder(tf.float32, [None, None, None, num_channels], name='rimage')
        targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

        snet = nf.create(limage, rimage, targets, FLAGS.net_type)

        lmap = tf.placeholder(tf.float32, [None, None, None, 64], name='lmap')
        rmap = tf.placeholder(tf.float32, [None, None, None, 64], name='rmap')


        map_prod = nf.map_inner_product(lmap, rmap)

        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(FLAGS.model_dir))

        for i in range(FLAGS.start_id, FLAGS.start_id + FLAGS.num_imgs):
            file_id = file_ids[i]
            print("FILE: ", file_ids[0], file_ids[1],file_ids[2],file_ids[3],file_ids[4])
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

            test_dict = {limage:linput, rimage:rinput, snet['is_training']: False}
            limage_map, rimage_map = session.run([snet['lbranch'], snet['rbranch']], feed_dict=test_dict)

            map_width = limage_map.shape[2]
            unary_vol = np.zeros((limage_map.shape[1], limage_map.shape[2], FLAGS.disp_range))
            for loc in range(FLAGS.disp_range):
                x_off = -loc
                l = limage_map[:, :, max(0, -x_off): map_width,:]
                r = rimage_map[:, :, 0: min(map_width, map_width + x_off),:]
                res = session.run(map_prod, feed_dict={lmap: l, rmap: r})
                #print("Inner product: ",res.shape)
                unary_vol[:, max(0, -x_off): map_width, loc] = res[0, :, :]
            if FLAGS.cost_aggregation:
              # We will use average pool 5x5 on original paper
              if FLAGS.average_pooling:
                print("Average pooling")
                unary_vol = unary_vol.reshape((1, unary_vol.shape[0], unary_vol.shape[1], unary_vol.shape[2]))
                unary_vol = nf.apply_cost_aggregation(unary_vol)
                unary_vol = tf.squeeze(unary_vol)
              # If not we will use semi global matching
              else:
                print("Semi-global matching")
                parameters = Parameters(max_disparity=FLAGS.disp_range, P1=10, P2=120, csize=(7, 7), bsize=(3, 3))
                paths = Paths()
                #print("here")
                unary_vol = aggregate_costs(unary_vol, parameters, paths)
                unary_vol = np.sum(unary_vol, axis=3)
            
            pred = tf.argmax(unary_vol, axis=2)
            # Convert tensor to array
            pred = pred.eval() * scale_factor
            # if FLAGS.post_processing:
            #   # Applying median filter size 3x3
            #   pred = cv2.medianBlur(pred,(3,3))

            misc.imsave('%s/disp_map_%06d_10.png' % (FLAGS.out_dir, file_id), pred)
            print('Image %s processed.' % (i + 1))
