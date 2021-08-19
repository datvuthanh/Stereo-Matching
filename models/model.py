from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization
import tensorflow as tf

def base_model(input_shape):
  inputs = Input(shape=input_shape)
  conv1 = Conv2D(32, (5,5),activation='relu',padding='valid')(inputs)
  bn1 = BatchNormalization()(conv1)
  conv2 = Conv2D(32, (5,5), activation='relu',padding='valid')(bn1)
  bn2 = BatchNormalization()(conv2)
  conv3 = Conv2D(32, (5,5), activation='relu',padding='valid')(bn2)
  bn3 = BatchNormalization()(conv3)
  conv4 = Conv2D(64, (5,5), activation='relu',padding='valid')(bn3)
  bn4 = BatchNormalization()(conv4)
  conv5 = Conv2D(64, (5,5), activation='relu',padding='valid')(bn4)
  bn5 = BatchNormalization()(conv5)
  conv6 = Conv2D(64, (5,5), activation='relu',padding='valid')(bn5)
  bn6 = BatchNormalization()(conv6)
  conv7 = Conv2D(64, (5,5), activation='relu',padding='valid')(bn6)
  bn7 = BatchNormalization()(conv7)
  conv8 = Conv2D(64, (5,5), activation='relu',padding='valid')(bn7)
  bn8 = BatchNormalization()(conv8)
  conv8 = Conv2D(64, (5,5), activation='relu',padding='valid')(bn7)
  bn8 = BatchNormalization()(conv8)
  conv9 = Conv2D(64, (5,5), activation=None,padding='valid')(bn8)
  bn9 = BatchNormalization()(conv9)
  model = Model(inputs=inputs, outputs=bn9)
  return model

def base_model_ws_9(input_shape): # Based on mc-cnn fast architecture
  inputs = Input(shape=input_shape)
  conv1 = Conv2D(64, (3,3),activation='relu',padding='valid')(inputs)
  bn1 = BatchNormalization()(conv1)
  conv2 = Conv2D(64, (3,3), activation='relu',padding='valid')(bn1)
  bn2 = BatchNormalization()(conv2)
  conv3 = Conv2D(64, (3,3), activation='relu',padding='valid')(bn2)
  bn3 = BatchNormalization()(conv3)
  conv4 = Conv2D(64, (3,3), activation=None,padding='valid')(bn3)
  bn4 = BatchNormalization()(conv4)
  model = Model(inputs=inputs, outputs=bn4)
  return model
