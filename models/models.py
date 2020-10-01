#from legos import ConvBNRelu
from keras.models import Model
from keras.layers import (
    Conv2D, BatchNormalization, Activation, Input,
    Multiply, Lambda, Activation)
import abc
import keras.backend as K

def ConvBNRelu(
        input_tensor,
        filters,
        kernel_size):

    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="valid")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def correlation_layer(x):
    '''
    Output must be of size 1 x half_range * 2 (e.g (1 x 200))
    According to the paper the feature layers in the bottom of the branches are:
        rbranch: 1 x 200 x 64
        lbranch: 1 x 1 x 64


    '''
    # Remove the downsampled spatial dimensions
    lbranch, rbranch = K.squeeze(x[0], 1), K.squeeze(x[1], 1)
    # Compute the output tensor
    rbranch = K.permute_dimensions(rbranch, (0, 2, 1))
    out_tensor = K.squeeze(K.batch_dot(lbranch, rbranch), 1)
    return out_tensor


class BaseModel(object):
    def __init__(self, left_input_shape, right_input_shape):
        self.left_input_shape = left_input_shape
        self.right_input_shape = right_input_shape

    def _create_branch(self):
        input = Input((None, None, 3))
        x = ConvBNRelu(
            input,
            self.num_feature_list[0],
            kernel_size=self.kernel_size)

        for l in range(1, self.num_layers-1, 1):
            x = ConvBNRelu(
                x,
                self.num_feature_list[l],
                kernel_size=self.kernel_size)

        x = Conv2D(
            self.num_feature_list[-1],
            kernel_size=self.kernel_size,
            activation="linear",
            padding="valid")(x)

        x = BatchNormalization()(x)
        x = Activation("softmax")(x)

        mdl = Model(inputs=input,
                    outputs=x)

        return mdl

    def build_model(self, add_corr_layer=True):
        input_left = Input(self.left_input_shape)
        input_right = Input(self.right_input_shape)

        base_branch = self._create_branch()

        # Reusing model weights for each branch
        out_left = base_branch(input_left)
        out_right = base_branch(input_right)

        if add_corr_layer:
            out = [Lambda(correlation_layer)([out_left, out_right])]
        else:
            out = [out_left, out_right]

        mdl = Model(
            inputs=[input_left, input_right],
            outputs=out)

        self.model = mdl


class dot_win19_dep9(BaseModel):
    def __init__(self, left_input_shape, right_input_shape):
        super(dot_win19_dep9, self).__init__(left_input_shape, right_input_shape)
        self.kernel_size = (3, 3)
        self.num_layers = 9
        self.num_feature_list = [64] * 9


class dot_win37_dep9(BaseModel):
    def __init__(self, left_input_shape, right_input_shape):
        super(dot_win37_dep9, self).__init__(left_input_shape, right_input_shape)
        self.kernel_size = (5, 5)
        self.num_layers = 9
        self.num_feature_list = [32] * 3 + [64] * 6
