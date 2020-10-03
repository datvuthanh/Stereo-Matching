import tensorflow as tf

#slim = tf.contrib.slim
import tf_slim as slim

def create_network(inputs, is_training, scope="win19_dep9", reuse=False):
	num_maps = 64
	kw = 3
	kh = 3

	with tf.variable_scope(scope, reuse=reuse):
		with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=tf.nn.relu, 
			normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
			
			net = slim.conv2d(inputs, num_maps, [kh, kw], scope='conv_bn_relu1')
			net = slim.repeat(net, 7, slim.conv2d, num_maps, [kh, kw], scope='conv_bn_relu2_8')
			net = slim.conv2d(net, num_maps, [kh, kw], scope='conv9', activation_fn=None, 
					normalizer_fn=None)
			net = slim.batch_norm(net, is_training=is_training)	

	return net
