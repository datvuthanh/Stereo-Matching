import tensorflow as tf
import os
import models.net_factory as nf
import numpy as np
from keras import backend as K
from data_handler import Data_handler


flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('num_iter', 40000, 'Total training iterations')
flags.DEFINE_string('model_dir', 'model', 'Trained network dir')
flags.DEFINE_string('data_version', 'kitti2012', 'kitti2012 or kitti2015')
flags.DEFINE_string('data_root', '/home/yinghui/stereoMatching/KITTI2012stereo/training', 'training dataset dir')
flags.DEFINE_string('util_root', '/home/yinghui/stereoMatching/preprocess/debug_12', 'Binary training files dir')
flags.DEFINE_string('net_type', 'win19_dep9', 'Network type: win37_dep9 pr win19_dep9')

flags.DEFINE_integer('eval_size', 200, 'number of evaluation patchs per iteration')
flags.DEFINE_integer('num_tr_img', 160, 'number of training images')
flags.DEFINE_integer('num_val_img', 34, 'number of evaluation images')
flags.DEFINE_integer('patch_size', 19, 'training patch size')
flags.DEFINE_integer('num_val_loc', 50000, 'number of validation locations')
flags.DEFINE_integer('disp_range', 201, 'disparity range')
flags.DEFINE_string('phase', 'train', 'train or evaluate')


FLAGS = flags.FLAGS

np.random.seed(123)

dhandler = Data_handler(data_version=FLAGS.data_version, 
	data_root=FLAGS.data_root,  
	util_root=FLAGS.util_root, 
	num_tr_img=FLAGS.num_tr_img, 
	num_val_img=FLAGS.num_val_img, 
	num_val_loc=FLAGS.num_val_loc, 
	batch_size=FLAGS.batch_size, 
	patch_size=FLAGS.patch_size, 
	disp_range=FLAGS.disp_range)


if FLAGS.data_version == 'kitti2012':
	num_channels = 1
elif FLAGS.data_version == 'kitti2015':
	num_channels = 3
else:
	sys.exit('data_version should be either kitti2012 or kitti2015')


def train():
	if not os.path.exists(FLAGS.model_dir):
		os.makedirs(FLAGS.model_dir)


	g = tf.Graph()
	with g.as_default():
		
		limage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, num_channels], name='limage')
		rimage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size + FLAGS.disp_range - 1, num_channels], name='rimage')
		targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

		snet = nf.create(limage, rimage, targets, FLAGS.net_type, FLAGS.patch_size, FLAGS.disp_range, FLAGS.data_version)

		loss = snet['loss']
		train_step = snet['train_step']
		session = tf.InteractiveSession()
		session.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep=1)

		acc_loss = tf.placeholder(tf.float32, shape=())
		loss_summary = tf.summary.scalar('loss', acc_loss)	
		train_writer = tf.summary.FileWriter(FLAGS.model_dir + '/training', g)

		saver = tf.train.Saver(max_to_keep=1)
		losses = []
		summary_index = 1
		lrate = 1e-2

		for it in range(1, FLAGS.num_iter):
			lpatch, rpatch, patch_targets = dhandler.next_batch()

			train_dict = {limage:lpatch, rimage:rpatch, targets:patch_targets, 
						snet['lrate']: lrate, K.learning_phase(): 1}
			_, mini_loss = session.run([train_step, loss], feed_dict=train_dict)
			losses.append(mini_loss)

			if it % 100 == 0:
				print('Loss at step: %d: %.6f' % (it, mini_loss))
				saver.save(session, os.path.join(FLAGS.model_dir, 'model.ckpt'), global_step=snet['global_step'])
				train_summary = session.run(loss_summary, 
					feed_dict={acc_loss: np.mean(losses)})
				train_writer.add_summary(train_summary, summary_index)
				summary_index += 1
				train_writer.flush()
				losses = []

			if it == 24000:
				lrate = lrate / 5.
			elif it > 24000 and (it - 24000) %  8000 == 0:
				lrate = lrate / 5.

def evaluate():
	lpatch, rpatch, patch_targets = dhandler.evaluate()
	labels = np.argmax(patch_targets, axis=1)

	with tf.Session() as session:

		limage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, num_channels], name='limage')
		rimage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size + FLAGS.disp_range - 1, num_channels], name='rimage')
		targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

		snet = nf.create(limage, rimage, targets, FLAGS.net_type, FLAGS.patch_size, FLAGS.disp_range, FLAGS.data_version)
		prod = snet['inner_product']
		predicted = tf.argmax(prod, axis=1)
		acc_count = 0

		saver = tf.train.Saver()
		saver.restore(session, tf.train.latest_checkpoint(FLAGS.model_dir))

		for i in range(0, lpatch.shape[0], FLAGS.eval_size):
			eval_dict = {limage:lpatch[i: i + FLAGS.eval_size], 
				rimage:rpatch[i: i + FLAGS.eval_size], K.learning_phase(): 0}
			pred = session.run([predicted], feed_dict=eval_dict)
			acc_count += np.sum(np.abs(pred - labels[i: i + FLAGS.eval_size]) <= 3)
			print('iter. %d finished, with %d correct (3-pixel error)' % (i + 1, acc_count))

		print('accuracy: %.3f' % ((acc_count / lpatch.shape[0]) * 100))


if FLAGS.phase == 'train':
	train()
elif FLAGS.phase == 'evaluate': 
	evaluate()
else:
	sys.exit('FLAGS.phase = train or evaluate')




