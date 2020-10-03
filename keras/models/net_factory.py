#import tensorflow as tf
import models.win19_dep9 as net19
import models.win37_dep9 as net37
from tensorflow.python.ops import control_flow_ops
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tf_slim as slim


def three_pixel_error(lbranch, rbranch, targets):
	lbranch2 = tf.squeeze(lbranch, [1])
	rbranch2 = tf.transpose(tf.squeeze(rbranch, [1]), perm=[0, 2, 1])
	prod = tf.matmul(lbranch2, rbranch2)
	prod_flatten = tf.layers.flatten(prod)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=prod_flatten), name='loss')
	return prod_flatten, loss

def create(limage, rimage, targets, net_type='win37_dep9'):
	is_training = tf.placeholder(tf.bool, [], name='is_training')
	with tf.name_scope('siamese_' + net_type):
		if net_type == 'win37_dep9':
			lbranch = net37.create_network(limage, is_training, reuse=False)
			rbranch = net37.create_network(rimage, is_training, reuse=True)

		elif net_type == 'win19_dep9':
			lbranch = net19.create_network(limage, is_training, reuse=False)
			rbranch = net19.create_network(rimage, is_training, reuse=True)
		else:
			sys.exit('Valid net_type: win37_dep9 or win19_dep9')

		prod_flatten, loss = three_pixel_error(lbranch, rbranch, targets)

		lrate = tf.placeholder(tf.float32, [], name='lrate')
		with tf.name_scope("optimizer"):
			global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)
			optimizer = tf.train.AdagradOptimizer(lrate)
			train_step = slim.learning.create_train_op(loss, optimizer, global_step=global_step)

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			if update_ops:
				updates = tf.group(*update_ops)
				loss = control_flow_ops.with_dependencies([updates], loss)

		net = {'lbranch': lbranch, 'rbranch': rbranch, 'loss': loss, 
			'inner_product': prod_flatten, 'train_step': train_step, 
			'is_training': is_training, 'global_step': global_step, 'lrate': lrate}

	return net


def map_inner_product(lmap, rmap):
	prod = tf.reduce_sum(tf.multiply(lmap, rmap), axis=3, name='map_inner_product')
	
	return prod
