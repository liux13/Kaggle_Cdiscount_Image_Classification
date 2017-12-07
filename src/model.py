import tensorflow as tf
import numpy as np
import time
import os

SAVE_DIR = r"/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/temp"
weight_path = os.path.join(SAVE_DIR, "vgg16_weights.npz")

class CNN_SiameseNet(object):
    """Implementation of the SiameseNet Using AlexNet Structure"""
    def __init__(self, left, right, weight_path=weight_path, trainable=True):
        """
        Create the computation graph of the SiameseNet
        :param left, right: Input tensors. Siamese requires a pair of input
        :param weight_path: Complete path to the pretrained weight file.
        """
        if weight_path is not None:
            self.data_dict = np.load(weight_path, encoding='latin1')
            print "Using saved weight:{}".format(weight_path)
        else:
            self.data_dict = None
            print "No weight initialization."
        self.trainable = trainable
        self.var_dict = {}

        # self.siamese_twins(left, right, train_mode=True, reuse=False)

    def build(self, left, right, reuse=False):
        assert left.get_shape().as_list() == right.get_shape().as_list()
        left_list = [tf.squezze(s, [1]) for s in tf.split(left, num_or_size_splits=4, axis=1)]


    def siamese_twins(self, left, right, train_mode=True, reuse=False):

        # First bottleneck
        conv1_1 = self._conv_layer(left, 3, 64, "conv1_1", reuse=reuse)              # 3x3 -> 224x224x64
        conv1_2 = self._conv_layer(conv1_1, 64, 64, "conv1_2", reuse=reuse)    # 3x3 -> 224x224x64
        pool1 = self._max_pool(conv1_2, 'pool1')                  # -> 112x112x64

        # Second bottleneck
        conv2_1 = self._conv_layer(pool1, 64, 128, "conv2_1", reuse=reuse)     # 3x3 -> 112x112x128
        conv2_2 = self._conv_layer(conv2_1, 128, 128, "conv2_2", reuse=reuse)  # 3x3 -> 112x112x128
        pool2 = self._max_pool(conv2_2, 'pool2')                  # -> 56x56x128

        # Third bottleneck
        conv3_1 = self._conv_layer(pool2, 128, 256, "conv3_1", reuse=reuse)    # 3x3 -> 56x56x256
        conv3_2 = self._conv_layer(conv3_1, 256, 256, "conv3_2", reuse=reuse)  # 3x3 -> 56x56x256
        conv3_3 = self._conv_layer(conv3_2, 256, 256, "conv3_3", reuse=reuse)  # 3x3 -> 56x56x256
        pool3 = self._max_pool(conv3_3, 'pool3')                  # -> 28x28x256

        # Fourth bottleneck
        conv4_1 = self._conv_layer(pool3, 256, 512, "conv4_1", reuse=reuse)    # 3x3 -> 28x28x512
        conv4_2 = self._conv_layer(conv4_1, 512, 512, "conv4_2", reuse=reuse)  # 3x3 -> 28x28x512
        conv4_3 = self._conv_layer(conv4_2, 512, 512, "conv4_3", reuse=reuse)  # 3x3 -> 28x28x512
        pool4 = self._max_pool(conv4_3, 'pool4')                  # -> 14x14x512

        # Fifth bottleneck
        conv5_1 = self._conv_layer(pool4, 512, 512, "conv5_1", reuse=reuse)    # 3x3 -> 14x14x512
        conv5_2 = self._conv_layer(conv5_1, 512, 512, "conv5_2", reuse=reuse)  # 3x3 -> 14x14x512
        conv5_3 = self._conv_layer(conv5_2, 512, 512, "conv5_3", reuse=reuse)  # 3x3 -> 14x14x512
        self.pool5 = self._max_pool(conv5_3, 'pool5')  # -> 7x7x512

        # # Global inference layer 1
        # self.fc6 = self._fc_layer(self.pool5, 25088, 4096, "fc6")           # 25088 -> 4096
        # self.relu6 = tf.nn.relu(self.fc6)
        #
        # # Global inference layer 2
        # self.fc7 = self._fc_layer(self.relu6, 4096, 4096, "fc7")            # 4096 -> 4096
        # self.relu7 = tf.nn.relu(self.fc7)
        #
        # # Instead of fc8, we have this layer output 4 numbers for regression
        # self.output = self._fc_layer(self.relu7, 4096, 4, name="output")    # 4096 -> 4 [num predictions]
        # self.data_dict = None


        return self.pool5

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    def _conv_layer(self, bottom, in_channels, out_channels, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            filters = self._get_var(name + "_W")
            conv_biases = self._get_var(name + "_b")

            conv = tf.nn.conv2d(input=bottom, filter=filters, strides=[1, 1, 1, 1], padding="SAME")
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def _fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            # weights, biases = self.get_var(in_size, out_size, name)
            weights = self._get_var(name + "_W")
            biases = self._get_var(name + "_b")

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def _get_var(self, name):
        assert isinstance(name, str)
        return tf.constant(self.data_dict.item()[name])


    def save_npz(self, sess, npy_path=os.path.join(SAVE_DIR, "CNN_Siamese.npz")):
        # assert isinstance(sess, tf.Session)
        timestr = "-" + time.strftime("%Y%m%d_%H%M")
        file_path = npy_path[:-4] + timestr + npy_path[-4:]
        data_dict = {}
        for name, var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name]= var_out

        np.savez(file_path, **data_dict)
        print "File saved as:", file_path

