# import tensorflow.keras as keras
from tensorflow import keras
import tensorflow as tf
import numpy as np
from Layer import GuideMap, Conv2D, FullConnect, SelfAttention
from Utils import apply_bg


class Inference_net(keras.Model):
    def __init__(self, config):
        super(Inference_net, self).__init__()
        self.guideMap = GuideMap()
        gd = config.luma_bins   # 8
        cm = config.channel_multiplier   # 3
        self.input_size = config.input_size  # 256
        self.cm = cm
        self.gd = gd
        spatial_bin = config.spatial_bin  # 16
        '''feature_collection'''
        self.n_ds_layers = int(np.log2(config.input_size / spatial_bin))
        ds_layer_list = []
        layer_nonorm = Conv2D(kernel_num=cm * gd, kernel_size=3, stride=2,
                              norm='bn', name='ds_layer{}'.format(0),
                              activation=config.activation)
        ds_layer_list.append(layer_nonorm)

        for i in range(1, self.n_ds_layers):
            layer = Conv2D(kernel_num=cm * gd * (2 ** i), kernel_size=3, stride=2,
                           norm='bn', name='ds_layer{}'.format(i),
                           activation=config.activation)
            ds_layer_list.append(layer)

        self.ds_layer = ds_layer_list
        self.SA = SelfAttention(filter_num=192,name='SA')
        '''global'''
        self.global_conv_1 = Conv2D(kernel_num=cm * gd * 8, kernel_size=3, stride=2,
                                    norm='bn', name='global_conv{}'.format(1),
                                    activation=config.activation)
        self.global_conv_2 = Conv2D(kernel_num=cm * gd * 8, kernel_size=3, stride=2,
                                    norm='bn', name='global_conv{}'.format(2),
                                    activation=config.activation)
        self.global_fc1 = FullConnect(32 * cm * gd, name='global_fc{}'.format(1), is_bn=True,
                                      activation=config.activation)
        self.global_fc2 = FullConnect(16 * cm * gd, name='global_fc{}'.format(2), is_bn=True,
                                      activation=config.activation)
        self.global_fc3 = FullConnect(8 * cm * gd, name='global_fc{}'.format(3), is_bn=False, activation='None')
        '''local'''
        self.local_conv_1 = Conv2D(kernel_num=cm * gd * 8, kernel_size=3, stride=1,
                                   norm='bn', name='local_conv{}'.format(1),
                                   activation=config.activation)
        self.local_conv_2 = Conv2D(kernel_num=cm * gd * 8, kernel_size=3, stride=1,
                                   norm=None, name='local_conv{}'.format(2),
                                   activation='None', use_bias=False)
        '''fusion'''
        self.fusion_conv = Conv2D(kernel_num=gd * 12, kernel_size=1, stride=1,
                                  norm=None, name='fusion_conv',
                                  activation='None', use_bias=True)

    def call(self, inputs, training=None, mask=None):

        tensor = tf.image.resize(inputs, (self.input_size, self.input_size))

        '''feature_collection'''
        for layer_op in self.ds_layer:
            tensor = layer_op(tensor)
        print('tensor', tensor)
        feature_collection = self.SA(tensor)

        print('feature_clooention',feature_collection)


        '''global'''
        tensor = self.global_conv_2(self.global_conv_1(feature_collection))
        shape = tensor.get_shape().as_list()
        # shape = tf.shape(tensor)
        tensor = tf.reshape(tensor, (shape[0], shape[1] * shape[2] * shape[3]))
        tensor = self.global_fc3(self.global_fc2(self.global_fc1(tensor)))
        global_feature = tensor

        '''local'''
        local_feature = self.local_conv_2(self.local_conv_1(feature_collection))
        '''fusion'''
        fusion_global = tf.reshape(global_feature, (shape[0], 1, 1, 8 * self.cm * self.gd))
        fusion_feature = tf.nn.relu(local_feature + fusion_global)
        grid = self.fusion_conv(fusion_feature)
        with tf.name_scope('unroll_grid'):
            grid = tf.stack(
                tf.split(grid, 3 * 4, axis=3), axis=4)
            grid = tf.stack(
                tf.split(grid, 4, axis=4), axis=5)

        '''guideMap'''
        print('input:',inputs)
        guide_map = self.guideMap(inputs)
        print('grid',grid)
        print('guide_map', guide_map)
        print('inputs', inputs)
        output = apply_bg(grid, guide_map, inputs)
        return output


