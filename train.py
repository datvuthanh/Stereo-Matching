import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import math
import cv2 
import datetime

# Cell 2: load images using Tensorflow

PATH = './datasets'

BUFFER_SIZE = 1024 * 4
BATCH_SIZE = 32  # for each positive and negative pairs, altogether = 32
n_train_samples = 117760


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 4

    rgb_pos = image[:, :w, :]
    nir_pos = image[:, w * 1:w * 2, :]
    rgb_neg = image[:, w * 2:w * 3, :]
    nir_neg = image[:, w * 3:w * 4, :]

    rgb_pos = tf.cast(rgb_pos, tf.float32)
    nir_pos = tf.cast(nir_pos, tf.float32)
    rgb_neg = tf.cast(rgb_neg, tf.float32)
    nir_neg = tf.cast(nir_neg, tf.float32)

    return rgb_pos, nir_pos, rgb_neg, nir_neg


# cell 3: data augmentation

def resize(input_l, input_r, target_l, target_r, height, width):
    input_l = tf.image.resize(input_l, [height, width],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_r = tf.image.resize(input_r, [height, width],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_l = tf.image.resize(target_l, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_r = tf.image.resize(target_r, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_l, input_r, target_l, target_r


def random_crop(input_l, input_r, target_l, target_r):

    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    stacked_image = tf.stack([input_l, input_r, target_l, target_r], axis=0)

    cropped_image = tf.image.random_crop(stacked_image, size=[4, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1], cropped_image[2], cropped_image[3]


# normalizing the images to [-1, 1]
def normalize(input_l, input_r, target_l, target_r):
    input_l = input_l / 255.0 # (input_l / 127.5) - 1
    input_r = input_r / 255.0 # (input_r / 127.5) - 1
    target_l = target_l / 255.0 #  (target_l / 127.5) - 1
    target_r = target_r / 255.0  #(target_r / 127.5) - 1

    return input_l, input_r, target_l, target_r


def random_jitter(input_l, input_r, target_l, target_r):
    # resize to 68x68
    input_l, input_r, target_l, target_r = resize(input_l, input_r, target_l, target_r, 68, 68)

    # crop
    input_l, input_r, target_l, target_r = random_crop(input_l, input_r, target_l, target_r)

    # flip_left_right
    if tf.random.uniform(()) > 0.5:
        input_l = tf.image.flip_left_right(input_l)
        input_r = tf.image.flip_left_right(input_r)
        target_l = tf.image.flip_left_right(target_l)
        target_r = tf.image.flip_left_right(target_r)

    # flip_up_down
    if tf.random.uniform(()) > 0.5:
        input_l = tf.image.flip_up_down(input_l)
        input_r = tf.image.flip_up_down(input_r)
        target_l = tf.image.flip_up_down(target_l)
        target_r = tf.image.flip_up_down(target_r)

    # brighness change
    if tf.random.uniform(()) > 0.5:
        rand_value = tf.random.uniform((), minval=-5.0, maxval=5.0)
        input_l = input_l + rand_value

        rand_value = tf.random.uniform((), minval=-5.0, maxval=5.0)
        input_r = input_r + rand_value

        rand_value = tf.random.uniform((), minval=-5.0, maxval=5.0)
        target_l = target_l + rand_value

        rand_value = tf.random.uniform((), minval=-5.0, maxval=5.0)
        target_r = target_r + rand_value

    # contrast change
    if tf.random.uniform(()) > 0.5:
        rand_value = tf.random.uniform((), minval=0.8, maxval=1.2)
        mean_value = tf.reduce_mean(input_l)
        input_l = (input_l - mean_value) * rand_value + mean_value

        rand_value = tf.random.uniform((), minval=0.8, maxval=1.2)
        mean_value = tf.reduce_mean(input_r)
        input_r = (input_r - mean_value) * rand_value + mean_value

        rand_value = tf.random.uniform((), minval=0.8, maxval=1.2)
        mean_value = tf.reduce_mean(target_l)
        target_l = (target_l - mean_value) * rand_value + mean_value

        rand_value = tf.random.uniform((), minval=0.8, maxval=1.2)
        mean_value = tf.reduce_mean(target_r)
        target_r = (target_r - mean_value) * rand_value + mean_value

    # clip value
    input_l = tf.clip_by_value(input_l, clip_value_min=0.0, clip_value_max=255.0)
    input_r = tf.clip_by_value(input_r, clip_value_min=0.0, clip_value_max=255.0)
    target_l = tf.clip_by_value(target_l, clip_value_min=0.0, clip_value_max=255.0)
    target_r = tf.clip_by_value(target_r, clip_value_min=0.0, clip_value_max=255.0)

    # rotate positive samples for making hard positive cases
    if tf.random.uniform(()) > 0.5:
        if tf.random.uniform(()) < 0.5:
            input_l = tfa.image.rotate(input_l, 1.5707963268)  # 90
            input_r = tfa.image.rotate(input_r, 1.570796326)  # 90
        else:
            input_l = tfa.image.rotate(input_l, 4.7123889804)  # 270
            input_r = tfa.image.rotate(input_r, 4.7123889804)  # 270

    return input_l, input_r, target_l, target_r


def load_image_train(image_file):
    input_l, input_r, target_l, target_r = load(image_file)
    input_l, input_r, target_l, target_r = random_jitter(input_l, input_r, target_l, target_r)
    input_l, input_r, target_l, target_r = normalize(input_l, input_r, target_l, target_r)

    return input_l, input_r, target_l, target_r


def load_image_test(image_file):
    input_l, input_r, target_l, target_r = load(image_file)
    input_l, input_r, target_l, target_r = resize(input_l, input_r, target_l, target_r, IMG_HEIGHT, IMG_WIDTH)
    input_l, input_r, target_l, target_r = normalize(input_l, input_r, target_l, target_r)

    return input_l, input_r, target_l, target_r

# cell 4: load training data

# train_dataset
train_dataset = tf.data.Dataset.list_files(PATH+'/*.png')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

def extract_first_features(filters, size, strides, apply_batchnorm=True):
    initializer = tf.keras.initializers.he_normal(seed=None)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
        result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.ReLU())

    return result


# cell 8: NIR domain matching

def base_model(input_x1, input_x2):
    x_1 = input_x1
    x_2 = input_x2

    # for x_1
    layer1 = extract_first_features(32, 3, 1, True)
    layer2 = extract_first_features(64, 3, 1, True)
    layer3 = extract_first_features(128, 3, 1, True)
    layer4 = extract_first_features(128, 5, 2, True)
    layer5 = extract_first_features(256, 3, 1, True)
    layer6 = extract_first_features(256, 5, 2, True)
    #layer7 = extract_first_features(256, 3, 1, True)
    #layer8 = extract_first_features(256, 5, 2, True)

    # for x_1
    x_1 = layer1(x_1)
    x_1 = layer2(x_1)
    x_1 = layer3(x_1)
    x_1 = layer4(x_1)
    x_1 = layer5(x_1)
    x_1 = layer6(x_1)
    #x_1 = layer7(x_1)
    #x_1 = layer8(x_1)
    x_1 = layers.Flatten()(x_1)

    # for x_2
    x_2 = layer1(x_2)
    x_2 = layer2(x_2)
    x_2 = layer3(x_2)
    x_2 = layer4(x_2)
    x_2 = layer5(x_2)
    x_2 = layer6(x_2)
    #x_2 = layer7(x_2)
    #x_2 = layer8(x_2)
    x_2 = layers.Flatten()(x_2)

    x = tf.abs(x_1 - x_2)
    #x = tf.concat([x_1, x_2, x], 1)

    return x


# cell 10: construct SPIMNet network

def make_similarity_model():
    inputs_1 = layers.Input(shape=[32, 32, 3])
    inputs_2 = layers.Input(shape=[32, 32, 3])
    left_patch = inputs_1
    right_patch = inputs_2

    # matching
    x = base_model(left_patch, right_patch)

    # concat features
    #x = tf.concat([f_nir, f_rgb], 1)

    # metric learning
    x = layers.Dense(1024)(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=[x])
    return model

similaritor = make_similarity_model()

# cell 13: build loss function

# path to save checkpoints
checkpoint_dir = './checkpoints/1201'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),similaritor=similaritor)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# Instantiate an optimizer.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.01)

def similaritor_loss(pos_output, neg_output):
    # total_loss1
    pos_loss = cross_entropy(tf.ones_like(pos_output), pos_output)
    neg_loss = cross_entropy(tf.zeros_like(neg_output), neg_output)

    total_loss = pos_loss + neg_loss

    return total_loss, pos_loss, neg_loss


# cell 14: train SPIMNet
log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(log_dir + "gan/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def train(train_data):
    for epoch in range(1, 50):
        start = time.time()
        print("EPOCH: ", epoch)
        # learning rate
        if epoch < 20:
            lr = 1e-3
        else:
            lr = 1e-4
        optimizer = tf.keras.optimizers.Adam(lr)
        
        checkpoint.step.assign_add(1)

        average_loss = 0
        average_posl = 0
        average_negl = 0
        average_l1lo = 0

        count = 0
        count_ones_pos = 0
        count_ones_neg = 0

        iter_samples = math.ceil(n_train_samples/BATCH_SIZE)
        progbar = tf.keras.utils.Progbar(iter_samples)

        for pos_bs_img0, pos_bs_img1, neg_bs_img0, neg_bs_img1 in train_data:
            progbar.update(count+1) # This will update the progress bar graph.
            #pos_bs_img1 = pos_bs_img1[:, :, :, 0:1]
            #neg_bs_img1 = neg_bs_img1[:, :, :, 0:1]

            with tf.GradientTape() as sim_tape:
                # training
                pos_output = similaritor([pos_bs_img0, pos_bs_img1], training=True)
                neg_output = similaritor([neg_bs_img0, neg_bs_img1], training=True)

                sim_loss, pos_loss, neg_loss = similaritor_loss(pos_output,neg_output)

                # --------- compute training acc ---------
                bool_pos_output = pos_output > 0
                ones_pos_output = tf.reduce_sum(tf.cast(bool_pos_output, tf.float32))
                count_ones_pos = count_ones_pos + ones_pos_output

                bool_neg_output = neg_output < 0
                ones_neg_output = tf.reduce_sum(tf.cast(bool_neg_output, tf.float32))
                count_ones_neg = count_ones_neg + ones_neg_output

            gradients = sim_tape.gradient(sim_loss, similaritor.trainable_variables)
            optimizer.apply_gradients(zip(gradients, similaritor.trainable_variables))
            
            
            if ( count % 1000 == 0):
              print('\t Loss at step: %d: Normal loss: %.6f, Negative Loss: %.6f, Sim loss: %.6f' % (int(count), pos_loss,neg_loss,sim_loss))

            average_loss = average_loss + sim_loss
            average_posl = average_posl + pos_loss
            average_negl = average_negl + neg_loss

            count = count + 1

        average_loss = average_loss / count
        average_posl = average_posl / count
        average_negl = average_negl / count

        print('epoch {}  average_loss {}  lr {}'.format(epoch, average_loss, lr))
        print('normal loss {}  perceptual loss {}'.format(average_posl, average_negl))

        pos_acc = (count_ones_pos * 100.0) / n_train_samples
        neg_acc = (count_ones_neg * 100.0) / n_train_samples
        print('train acc (pos) {} - acc (neg) {}'.format(pos_acc, neg_acc))
        print()

        with summary_writer.as_default():
            tf.summary.scalar('average_loss', average_loss, step=epoch)
            tf.summary.scalar('pos_loss', average_posl, step=epoch)
            tf.summary.scalar('neg_loss', average_negl, step=epoch)
            tf.summary.scalar('pos_acc', pos_acc, step=epoch)
            tf.summary.scalar('neg_acc', neg_acc, step=epoch)

        save_path = manager.save()
        print("\n----------Saved checkpoint for epoch {}: {}-----------\n".format(epoch+1, save_path))
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,time.time()-start))

# cell 15: train SPIMNet with 35 epochs
checkpoint.restore(manager.latest_checkpoint)
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")
train(train_dataset)
