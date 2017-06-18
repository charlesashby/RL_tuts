from lib.inits import fc_initializer, conv_initializer
import tensorflow as tf


def conv2d(input_, output_dim, k_h, k_w, stride=1, name="conv", activation=tf.nn.relu):
    """ 2D Convolutional Layer """

    with tf.variable_scope(name):
        input_channels = int(input_.get_shape()[-1])
        W = tf.get_variable('W', [k_h, k_w, input_channels, output_dim],
                            initializer=conv_initializer(k_w, k_h, input_channels))
        b = tf.get_variable('b', [output_dim],
                            initializer=conv_initializer(k_w, k_h, input_channels))

        preact = tf.nn.conv2d(input_, W, strides=[1, stride, stride, 1],
                              padding='VALID') + b

        if activation:
            return activation(preact)
        else:
            return preact

def fc_layer(input_, out_dim, name=None, activation=tf.nn.relu):
    """ Fully-Connected Layer """

    with tf.variable_scope(name or 'fc'):
        input_channels = int(input_.get_shape()[1])
        W = tf.get_variable('W', [input_channels, out_dim],
                            initializer=fc_initializer(input_channels))
        b = tf.get_variable('b', [out_dim],
                            initializer=fc_initializer(input_channels))

    preact = tf.nn.softmax(tf.matmul(input_, W) + b)

    if activation:
        return activation(preact)
    else:
        return preact