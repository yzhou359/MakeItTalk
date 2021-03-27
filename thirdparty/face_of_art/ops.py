import tensorflow as tf
import numpy as np


def conv_relu_pool(input, conv_ker, conv_filters, conv_stride=1, conv_padding='SAME',
                   conv_ker_init=tf.random_normal_initializer(0.01), conv_bias_init=tf.zeros_initializer(),
                   pool_size=2, pool_stride=2, pool_padding='same', var_scope='layer', reuse=None):

    with tf.variable_scope(var_scope):
        conv = tf.layers.conv2d(input, filters=conv_filters, kernel_size=[conv_ker, conv_ker],
                                strides=conv_stride, padding=conv_padding, bias_initializer=conv_bias_init,
                                kernel_initializer=conv_ker_init, name='conv', reuse=reuse)
        relu = tf.nn.relu(conv, name='relu')
        out = tf.layers.max_pooling2d(relu, pool_size=(pool_size, pool_size),
                                      strides=(pool_stride, pool_stride), padding=pool_padding, name='pool')
        return out


def conv_relu(input, conv_ker, conv_filters, conv_stride=1, conv_dilation=1, conv_padding='SAME',
              conv_ker_init=tf.random_normal_initializer(0.01), conv_bias_init=tf.zeros_initializer(),
              var_scope='layer', reuse=None):

    with tf.variable_scope(var_scope):
        conv = tf.layers.conv2d(input, filters=conv_filters, kernel_size=[conv_ker, conv_ker],
                                strides=conv_stride, dilation_rate=conv_dilation, padding=conv_padding,
                                bias_initializer=conv_bias_init, kernel_initializer=conv_ker_init, name='conv',
                                reuse=reuse)
        out = tf.nn.relu(conv, name='relu')
        return out


def conv(input, conv_ker, conv_filters, conv_stride=1, conv_dilation=1, conv_padding='SAME',
         conv_ker_init=tf.random_normal_initializer(0.01), conv_bias_init=tf.zeros_initializer(),
         var_scope='layer', reuse=None):

    with tf.variable_scope(var_scope):
        out = tf.layers.conv2d(input, filters=conv_filters, kernel_size=[conv_ker, conv_ker],
                               strides=conv_stride, dilation_rate=conv_dilation, padding=conv_padding,
                               bias_initializer=conv_bias_init, kernel_initializer=conv_ker_init, name='conv',
                               reuse=reuse)
        return out


def deconv(input, conv_ker, conv_filters, conv_stride=1, conv_padding='SAME',
           conv_ker_init=tf.random_normal_initializer(0.01), conv_bias_init=tf.zeros_initializer(),
           var_scope='layer', reuse=None):

    with tf.variable_scope(var_scope):
        out = tf.layers.conv2d_transpose(input, filters=conv_filters, kernel_size=[conv_ker, conv_ker],
                                         strides=conv_stride, padding=conv_padding, bias_initializer=conv_bias_init,
                                         kernel_initializer=conv_ker_init, name='deconv', reuse=reuse)
        return out


def deconv2d_bilinear_upsampling_initializer(shape):
    """Returns the initializer that can be passed to DeConv2dLayer for initializ ingthe
    weights in correspondence to channel-wise bilinear up-sampling.
    Used in segmentation approaches such as [FCN](https://arxiv.org/abs/1605.06211)
    Parameters
    ----------
    shape : tuple of int
        The shape of the filters, [height, width, output_channels, in_channels].
        It must match the shape passed to DeConv2dLayer.
    Returns
    -------
    ``tf.constant_initializer``
        A constant initializer with weights set to correspond to per channel bilinear upsampling
        when passed as W_int in DeConv2dLayer
    --------
    from: tensorlayer
    https://github.com/tensorlayer/tensorlayer/blob/c7a1a4924219244c71048709ca729aca0c34c453/tensorlayer/layers/convolution.py
    """
    if shape[0] != shape[1]:
        raise Exception('deconv2d_bilinear_upsampling_initializer only supports symmetrical filter sizes')
    if shape[3] < shape[2]:
        raise Exception('deconv2d_bilinear_upsampling_initializer behaviour is not defined for num_in_channels < num_out_channels ')

    filter_size = shape[0]
    num_out_channels = shape[2]
    num_in_channels = shape[3]

    # Create bilinear filter kernel as numpy array
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                                    (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_out_channels, num_in_channels))
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    # assign numpy array to constant_initalizer and pass to get_variable
    bilinear_weights_init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return bilinear_weights_init