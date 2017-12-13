import tensorflow as tf
import numpy as np
import time
import os

SAVE_DIR = r"/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/temp"
weight_path = os.path.join(SAVE_DIR, "vgg16_weights.npz")
IMG_DIM = 224
IMG_CHNL = 3

from src import datagenerator
from tensorflow.contrib.data import Iterator

DATA_DIR = "/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/data/"
SAVE_DIR = r"/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/temp"
weight_path = os.path.join(SAVE_DIR, "vgg16_weights.npz")

train_file_path = os.path.join(DATA_DIR, "train_images.csv")
val_file_path = os.path.join(DATA_DIR, "val_images.csv")
# batch_size = 10

class CNN_SiameseNet(object):
    """Implementation of the SiameseNet Using AlexNet Structure"""
    def __init__(self, batch_size, weight_path=weight_path, trainable=True):
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
        self.batch_size = batch_size

        # self.siamese_twins(left, right, train_mode=True, reuse=False)

    def get_valid_indexes(self, batch_size, num_arr):
        prod_idxs = range(0, 4 * batch_size, 4)
        valid_indexes = []
        for prod, size in enumerate(num_arr):
            for i in range(size):
                valid_indexes.append(prod_idxs[prod] + i)
        return np.array(valid_indexes).astype(np.int32)

    def build(self, left, left_mask, right=None, right_mask=None, y=None, margin=0.2, keep_prob=0.5):
        print "Model building started."

        assert (not right) == (not right_mask) == (not y)
        inference_mode = not y

        left.set_shape([None, 4, IMG_DIM, IMG_DIM, IMG_CHNL])
        left_mask.set_shape([None, 4])

        if not inference_mode:
            right.set_shape([None, 4, IMG_DIM, IMG_DIM, IMG_CHNL])
            right_mask.set_shape([None, 4])
            y.set_shape([None, ])

        start_time = time.time()
        self.o1 = self.siamese_twins(left, left_mask, train_mode=self.trainable, first_run=True, keep_prob=keep_prob)

        if not inference_mode:
            self.o2 = self.siamese_twins(
                right, right_mask, train_mode=self.trainable, first_run=False, keep_prob=keep_prob)
            assert self.o1.get_shape().as_list() == self.o2.get_shape().as_list()

            self.loss = self.loss_with_spring(self.o1, self.o2, y, margin=5)
            self.r_loss = self.loss
        print "Model building finished: %ds" %(time.time() - start_time)

    def siamese_twins(self, x, mask, train_mode=True, first_run=True, keep_prob=0.5):
        x  = tf.boolean_mask(x, mask)
        x = tf.reshape(x, (-1, IMG_DIM, IMG_DIM, IMG_CHNL))

        self.num_imgs = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        self.num_imgs.set_shape([None])

        # First bottleneck
        conv1_1 = self._conv_layer(x, 3, 64, "conv1_1")           # 3x3 -> 224x224x64
        conv1_2 = self._conv_layer(conv1_1, 64, 64, "conv1_2")    # 3x3 -> 224x224x64
        pool1 = self._max_pool(conv1_2, 'pool1')                  # -> 112x112x64

        # Second bottleneck
        conv2_1 = self._conv_layer(pool1, 64, 128, "conv2_1")     # 3x3 -> 112x112x128
        conv2_2 = self._conv_layer(conv2_1, 128, 128, "conv2_2")  # 3x3 -> 112x112x128
        pool2 = self._max_pool(conv2_2, 'pool2')                  # -> 56x56x128

        # Third bottleneck
        conv3_1 = self._conv_layer(pool2, 128, 256, "conv3_1")    # 3x3 -> 56x56x256
        conv3_2 = self._conv_layer(conv3_1, 256, 256, "conv3_2")  # 3x3 -> 56x56x256
        conv3_3 = self._conv_layer(conv3_2, 256, 256, "conv3_3")  # 3x3 -> 56x56x256
        pool3 = self._max_pool(conv3_3, 'pool3')                  # -> 28x28x256

        # Fourth bottleneck
        conv4_1 = self._conv_layer(pool3, 256, 512, "conv4_1")    # 3x3 -> 28x28x512
        conv4_2 = self._conv_layer(conv4_1, 512, 512, "conv4_2")  # 3x3 -> 28x28x512
        conv4_3 = self._conv_layer(conv4_2, 512, 512, "conv4_3")  # 3x3 -> 28x28x512
        pool4 = self._max_pool(conv4_3, 'pool4')                  # -> 14x14x512

        # Fifth bottleneck
        conv5_1 = self._conv_layer(pool4, 512, 512, "conv5_1")    # 3x3 -> 14x14x512
        conv5_2 = self._conv_layer(conv5_1, 512, 512, "conv5_2")  # 3x3 -> 14x14x512
        conv5_3 = self._conv_layer(conv5_2, 512, 512, "conv5_3")  # 3x3 -> 14x14x512
        pool5 = self._max_pool(conv5_3, 'pool5')                  # -> 7x7x512

        # Inference layer1
        fc6 = self._fc_layer(pool5, 25088, 4096, "fc6")
        relu6 = tf.nn.relu(fc6)
        if train_mode:
            relu6 = tf.nn.dropout(relu6, keep_prob)

        # Inference layer2
        fc7 = self._fc_layer(relu6, 4096, 4096, "fc7")
        relu7 = tf.nn.relu(fc7)
        if train_mode:
            relu7 = tf.nn.dropout(relu7, keep_prob)

        # Inference layer3
        fc_8 = self._fc_layer(relu7, 4096, 128, "fc_8")

        # compute product coordinates
        with tf.name_scope("embedding"):
            embedding = tf.stack([tf.reduce_max(em, axis=0)
                                  for em in tf.split(fc_8, num_or_size_splits=self.num_imgs, num=self.batch_size)],
                                 axis=0)
        return embedding


    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    def _conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            filters, conv_biases = self._get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(input=bottom, filter=filters, strides=[1, 1, 1, 1], padding="SAME")
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def _get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self._get_var(initial_value, name + "_W")

        initial_value = tf.truncated_normal([out_channels], 0.0, .001)
        biases = self._get_var(initial_value, name + "_b")

        return filters, biases

    def _fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self._get_var(initial_value, name + "_W")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self._get_var(initial_value, name + "_b")
        return weights, biases


    def _get_var(self, initial_value, name):
        assert isinstance(name, str)
        if self.data_dict is not None and name in self.data_dict.keys():
            value = self.data_dict[name]
        else:
            value = initial_value

        if self.trainable:
            var = tf.get_variable(name=name, initializer=value)
        else:
            var = tf.constant(value, dtype=tf.float32, name=name)

        self.var_dict[name] = var

        assert var.get_shape() == initial_value.get_shape()

        return var


    def save_npz(self, sess, npy_path=os.path.join(SAVE_DIR, "CNN_Siamese.npz")):
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

    def loss_with_spring(self, o1, o2, y, margin):
        margin = 5.0
        with tf.name_scope("loss_with_spring"):
            labels_t = tf.cast(y, tf.float32)
            labels_f = tf.subtract(1.0, labels_t, name="1-yi")

            eucd2 = tf.pow(tf.subtract(o1, o2), 2)
            eucd2 = tf.reduce_sum(eucd2, 1)
            eucd = tf.sqrt(eucd2 + 1e-6, name="eucd")

            C = tf.constant(margin, name="C")
            pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
            neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")

            losses = tf.add(pos, neg, name="losses")
            loss = tf.reduce_mean(losses, name="loss", axis=0)
            return loss

    # def contrastive_loss(self, model1, model2, y, margin):
    #     with tf.name_scope("contrastive-loss"):
    #         d = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keep_dims=True))
    #         print 'd', d.get_shape()
    #         print 'y', y.get_shape()
    #         tmp = tf.cast(y, tf.float32) * tf.square(d)
    #         print 'tmp', tmp.get_shape()
    #         tmp2 = tf.cast(1 - y, tf.float32) * tf.square(tf.maximum((margin - d), 0))
    #         print 'tmp2', tmp2.get_shape()
    #         out = tf.reduce_mean(tmp + tmp2) / 2
    #         print 'out', out.get_shape()
    #         return out


# if __name__ == "__main__":
#     tf.reset_default_graph()
#
#     with tf.device("/cpu:0"):
#         tr_data = datagenerator.ImageDataGenerator(sample_file_path=train_file_path,
#                                                    shuffle=True,
#                                                    is_training=True,
#                                                    batch_size=batch_size)
#         val_data = datagenerator.ImageDataGenerator(sample_file_path=val_file_path,
#                                                     shuffle=True,
#                                                     is_training=False,
#                                                     batch_size=batch_size)
#
#         # create an reinitializable iterator given the dataset structure
#         iterator = Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
#         next_batch = iterator.get_next()
#
#         training_init_op = iterator.make_initializer(tr_data.data)
#         validation_init_op = iterator.make_initializer(val_data.data)
#
#         # graph input
#         left, left_mask, right, right_mask, Y, left_labels, right_labels = next_batch
#
#         global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
#
#         # Initialize model
#         siamese = CNN_SiameseNet(batch_size=batch_size, trainable=True)
#         siamese.build(left, left_mask, right, right_mask, Y, margin=.2, keep_prob=0.5)
#
#         #     train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(siamese.loss,
#         #                                                                                     global_step=global_step)
#         train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
#
#         with tf.Session() as sess:
#             # Initialize an iterator over a dataset with 10 elements.
#             sess.run(tf.global_variables_initializer())
#             sess.run(training_init_op)
#             # sess.run(validation_init_op)
#             y_, o1, o2, loss_, _ = sess.run([Y, siamese.o1, siamese.o2, siamese.loss, train_step])