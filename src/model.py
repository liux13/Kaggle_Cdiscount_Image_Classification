import tensorflow as tf
import numpy as np


weight_path = r"/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/src/vgg16_weights.npz"

class CNN_SiameseNet(object):
    """Implementation of the SiameseNet Using AlexNet Structure"""
    def __init__(self, x1, x2, weight_path=weight_path):
        """
        Create the computation graph of the SiameseNet
        :param x: Placeholder for the input tensor
        :param weight_path: Complete path to the pretrained weight file.
        """
        self.data_dict = np.load(weight_path, encoding='latin1')

        assert len(x1.get_shape()) == 5
        self.build_siamese_twins(x1[0])


    def build_siamese_twins(self, x):

        # First bottleneck
        self.conv1_1 = self._conv_layer(x, 3, 64, "conv1_1")              # 3x3 -> 224x224x64
        self.conv1_2 = self._conv_layer(self.conv1_1, 64, 64, "conv1_2")    # 3x3 -> 224x224x64
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')                  # -> 112x112x64

        # Second bottleneck
        self.conv2_1 = self._conv_layer(self.pool1, 64, 128, "conv2_1")     # 3x3 -> 112x112x128
        self.conv2_2 = self._conv_layer(self.conv2_1, 128, 128, "conv2_2")  # 3x3 -> 112x112x128
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')                  # -> 56x56x128

        # Third bottleneck
        self.conv3_1 = self._conv_layer(self.pool2, 128, 256, "conv3_1")    # 3x3 -> 56x56x256
        self.conv3_2 = self._conv_layer(self.conv3_1, 256, 256, "conv3_2")  # 3x3 -> 56x56x256
        self.conv3_3 = self._conv_layer(self.conv3_2, 256, 256, "conv3_3")  # 3x3 -> 56x56x256
        self.pool3 = self._max_pool(self.conv3_3, 'pool3')                  # -> 28x28x256

        # Fourth bottleneck
        self.conv4_1 = self._conv_layer(self.pool3, 256, 512, "conv4_1")    # 3x3 -> 28x28x512
        self.conv4_2 = self._conv_layer(self.conv4_1, 512, 512, "conv4_2")  # 3x3 -> 28x28x512
        self.conv4_3 = self._conv_layer(self.conv4_2, 512, 512, "conv4_3")  # 3x3 -> 28x28x512
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')                  # -> 14x14x512

        # Fifth bottleneck
        self.conv5_1 = self._conv_layer(self.pool4, 512, 512, "conv5_1")    # 3x3 -> 14x14x512
        self.conv5_2 = self._conv_layer(self.conv5_1, 512, 512, "conv5_2")  # 3x3 -> 14x14x512
        self.conv5_3 = self._conv_layer(self.conv5_2, 512, 512, "conv5_3")  # 3x3 -> 14x14x512
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')  # -> 7x7x512

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

        # print "Model building finished: %ds" % (time.time() - start_time)
        return self.output

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    def _conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
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

