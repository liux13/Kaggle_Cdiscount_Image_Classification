import os
import io
from collections import defaultdict
import bson
import tensorflow as tf
import numpy as np
import pandas as pd
from skimage.data import imread
from skimage.exposure import rescale_intensity
from skimage.transform import rotate
from tensorflow import data

DATADIR = r"/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/data/"
VGG_MEAN = np.array([103.939, 116.779, 123.68]).reshape((1,1,3))    #VGG_MEAN in bgr order
offset_path = os.path.join(DATADIR, "train_offsets.csv")
record_defaults = np.ones((8, 1)).tolist()
IMG_DIM = 180
OUT_DIM = 224
IMG_CHNL = 3

class ImageDataGenerator(object):
    """
    Wrapper class around the new Tensorflow's dataset pipeline.
    
    """
    def __init__(self,
                 sample_file_path,
                 offset_file_path=offset_path,
                 is_training=True,
                 batch_size=2,
                 shuffle=True,
                 buffer_size=500,
                 same_prob=0.5,
                 inference_mode=False):
        """Creates a new ImageDataGenerator.
        
        Receives a path string to a text file, which consists of many lines, where each line specifies the relative
        location of an image. Using this data, this class will create TensorFlow dataset that can be used to train
        rectifynet.
        
        :param sample_file_path: Path to the sample csv file
        :param offset_path: Path to the offset file for record random retrieval
        :param mode: A boolean value indicating "train" or "validation" status. Depending on this value, pre-processing
            is done differently.
        :param batch_size: Number of images per batch.
        :param shuffle: Whether or not to shuffle the data in the dataset and the initial file list.
        :param buffer_size: Number of image dirs used as buffer for TensorFlows shuffling of the dataset. 
            If not specified, the entire txt_file will be buffered into memory for shuffling.
        """
        self.sample_file_path = sample_file_path
        self.offset_file_path = offset_file_path
        self.is_training = is_training
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.same_prob = same_prob
        self.prod_indices = range(batch_size)
        self.inference_mode = inference_mode
        if not self.inference_mode:
            self._read_csv_file()
        dataset = data.TextLineDataset(sample_file_path).skip(1)

        if shuffle and not self.inference_mode:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        if not self.inference_mode:
            dataset = dataset.map(lambda row: tf.py_func(self._data_augment,
                                                         [row, True],
                                                         [tf.float32, tf.bool, tf.float32,
                                                          tf.bool, tf.int32, tf.int16, tf.int16]))
        else:
            dataset = dataset.map( lambda row: tf.py_func(self._data_augment,
                                                        [row, False],
                                                        [tf.float32, tf.bool, tf.int32, tf.int32]))

        self.data = dataset.batch(self.batch_size)


    def _read_csv_file(self):
        print "Making look up table..."
        self._dataframe = pd.read_csv(self.sample_file_path, index_col=0)
        self._lvl3_dict = defaultdict(list)         #{label: product_ids}
        for ir in self._dataframe.itertuples():
            self._lvl3_dict[ir[-1]].append(ir[0])

    def _get_pair_product_id(self, label, isSame):

        if not isSame:
            label_ = np.random.choice(self._lvl3_dict.keys())
            while (label_ == label):
                label_ = np.random.choice(self._lvl3_dict.keys())
            label = label_

        product_id = np.random.choice(self._lvl3_dict[label], size=1)[0]
        row = self._dataframe[self._dataframe.index==product_id]

        return row, label


    def _data_augment(self, row, find_pair=False):
        """
           This function takes a single row of df as input, warp it randomly and returns 4 numbers as targets
        """
        if isinstance(row, str):
            cols = [int(x) for x in row.split(',')]
            product_id = cols[0]
            num_imgs = cols[1]
            offset = cols[2]
            length = cols[3]
            label = cols[-1]
        else:
            num_imgs, offset, length, _, _, _, label = row.values[0]

        x1 = np.random.rand(4, OUT_DIM, OUT_DIM, IMG_CHNL)*255.
        mask = np.zeros((4, )).astype(bool)

        with open(os.path.join(DATADIR, "train.bson")) as b:
            b.seek(offset)
            sample = b.read(length)
            item = bson.BSON(sample).decode()

            for i in range(num_imgs):
                mask[i] = True
                pic = imread(io.BytesIO(item['imgs'][i]['picture']))

                # rescale and padding
                b_pic = np.random.random(size=(2 * IMG_DIM, 2 * IMG_DIM, 3)) * 255.
                center = np.array(b_pic.shape[:2]) / 2
                b_pic[center[0] - IMG_DIM / 2:center[0] + IMG_DIM / 2,
                center[1] - IMG_DIM / 2:center[1] + IMG_DIM / 2] = pic

                if self.is_training:
                    # rotate
                    angle = np.random.randint(20)
                    b_pic = rotate(b_pic, angle, mode='reflect')

                    # flip
                    dice = np.random.random(1)
                    if dice > 0.75:
                        b_pic = np.fliplr(b_pic)
                    elif dice > 0.5:
                        b_pic = np.flipud(b_pic)

                    b_pic = rescale_intensity(
                        b_pic,
                        in_range='image',
                        out_range=(np.random.randint(0, 127),
                        np.random.randint(127, 255))
                    )

                # cropping to correct output dimension
                cropped = b_pic[center[0] - OUT_DIM / 2:center[0] + OUT_DIM / 2,
                          center[1] - OUT_DIM / 2:center[1] + OUT_DIM / 2]
                # bgr and subtract mean
                cropped = cropped[:,:,::-1]-VGG_MEAN
                x1[i, ...] = cropped

        if find_pair:
            isSame = np.random.random(size=1) > self.same_prob
            row, _label = self._get_pair_product_id(label, isSame=isSame[0])
            x2, mask2, num_imgs2, _ = self._data_augment(row, find_pair=False)
            return x1.astype(np.float32), mask, x2, mask2, np.int32(isSame[0]), np.int16(label), np.int16(_label)


        return x1.astype(np.float32), mask, np.int32(num_imgs), np.int32(label)
        # return x1.astype(np.float32), mask, np.int32(label), np.int32(product_id)