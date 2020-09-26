import abc

from keras.backend import permute_dimensions, squeeze, batch_dot
from keras.layers import Activation, BatchNormalization, Conv2D, Input, Lambda
from keras.models import Model, Sequential


def correlation_layer(x):
    lbranch, rbranch = squeeze(x[0], 1), squeeze(x[1], 1)
    rbranch = permute_dimensions(rbranch, (0, 2, 1))
    out_tensor = squeeze(batch_dot(lbranch, rbranch), 1)
    return out_tensor


def build_branch(kernel_size, num_layers):
    model = Sequential()
    for i in range(num_layers):
        model.add(Conv2D(64, kernel_size, activation='relu'))
        model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model


def build_model(left_input_shape, right_input_shape):
    input_left = Input(left_input_shape)
    input_right = Input(right_input_shape)

    base_branch = build_branch(kernel_size=(5, 5), num_layers=9)
    out_left = base_branch(input_left)
    out_right = base_branch(input_right)

    out = [Lambda(correlation_layer)([out_left, out_right])]
    model = Model(inputs=[input_left, input_right], outputs=out)
    return model
