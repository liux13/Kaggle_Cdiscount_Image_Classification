import tensorflow as tf
import numpy as np
import time
import os

SAVE_DIR = r"/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/temp"
weight_path = os.path.join(SAVE_DIR, "vgg16_weights.npz")

class CNN_SiameseNet(object):
    """Implementation of the SiameseNet Using AlexNet Structure"""
    def __init__(self, weight_path=weight_path, trainable=True):
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

    def build(self, left, right, y, margin=0.2, keep_prob=0.5):
        assert left.get_shape().as_list() == right.get_shape().as_list()

        self.o1 = self.siamese_twins(left, train_mode=self.trainable, first_run=True, keep_prob=keep_prob)
        self.o2 = self.siamese_twins(right, train_mode=self.trainable, first_run=False, keep_prob=keep_prob)        # (None, 128)
        assert self.o1.get_shape().as_list() == self.o2.get_shape().as_list()

        self.loss = self.contrastive_loss(self.o1, self.o2, y, margin)



    def siamese_twins(self, x, train_mode=True, first_run=True, keep_prob=0.5):
        features = []
        for i, batch_imgs in enumerate([tf.squeeze(s, [1]) for s in tf.split(x, num_or_size_splits=4, axis=1)]):
            reuse = False if i == 0 and first_run else True

            # First bottleneck
            conv1_1 = self._conv_layer(batch_imgs, 3, 64, "conv1_1", reuse=reuse)              # 3x3 -> 224x224x64
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
            pool5 = self._max_pool(conv5_3, 'pool5')  # -> 7x7x512

            # Inference layer1
            fc6 = self._fc_layer(pool5, 25088, 4096, "fc6", reuse=reuse)
            relu6 = tf.nn.relu(fc6)
            if train_mode:
                relu6 = tf.nn.dropout(relu6, keep_prob)

            # Inference layer2
            fc7 = self._fc_layer(relu6, 4096, 4096, "fc7", reuse=reuse)
            relu7 = tf.nn.relu(fc7)
            if train_mode:
                relu7 = tf.nn.dropout(relu7, keep_prob)

            # Inference layer3
            embedding = tf.layers.dense(relu7, 128, use_bias=True, name="embedding", reuse=reuse)
            # embedding = self._fc_layer(relu7, 4096, 128, "embedding", reuse=reuse)

            features.append(embedding)

        features = tf.stack([tf.expand_dims(t, axis=1) for t in features], 1)

        return features

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


    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    def _conv_layer(self, bottom, in_channels, out_channels, name, reuse):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            filters = self._get_var(name + "_W")
            conv_biases = self._get_var(name + "_b")

            conv = tf.nn.conv2d(input=bottom, filter=filters, strides=[1, 1, 1, 1], padding="SAME")
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def _fc_layer(self, bottom, in_size, out_size, name, reuse=False):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # weights, biases = self.get_var(in_size, out_size, name)
            weights = self._get_var(name + "_W")
            biases = self._get_var(name + "_b")

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def _get_var(self, name):
        assert isinstance(name, str)
        return tf.get_variable(name=name, initializer=self.data_dict[name])



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

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def contrastive_loss(self, model1, model2, y, margin):
        with tf.name_scope("contrastive-loss"):
            d = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keep_dims=True))
            tmp = tf.cast(y, tf.float32) * tf.square(d)
            tmp2 = tf.cast(1 - y, tf.float32) * tf.square(tf.maximum((margin - d), 0))
            return tf.reduce_mean(tmp + tmp2) / 2


    # def loss_with_step(self):
    #     margin = 5.0
    #     labels_t = self.y_
    #     labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
    #     eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
    #     eucd2 = tf.reduce_sum(eucd2, 1)
    #     eucd = tf.sqrt(eucd2+1e-6, name="eucd")
    #     C = tf.constant(margin, name="C")
    #     pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
    #     neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
    #     losses = tf.add(pos, neg, name="losses")
    #     loss = tf.reduce_mean(losses, name="loss")
    #     return loss

