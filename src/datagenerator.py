import os
import tensorflow as tf
import numpy as np
from skimage import img_as_float

from tensorflow import data
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

DATADIR = r"/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/data/"
VGG_MEAN = np.array([103.939, 116.779, 123.68]).reshape((1,1,3))    #VGG_MEAN in bgr order
offset_path = os.path.join(DATADIR, "categories.csv")

class ImageDataGenerator(object):
    """
    Wrapper class around the new Tensorflow's dataset pipeline.
    
    """
    def __init__(self, csv_file_path, offset_file_path=offset_path, is_training=True, batch_size=128, shuffle=True, buffer_size=None):
        """Creates a new ImageDataGenerator.
        
        Receives a path string to a text file, which consists of many lines, where each line specifies the relative
        location of an image. Using this data, this class will create TensorFlow dataset that can be used to train
        rectifynet.
        
        :param csv_file_path: Path to the text file
        :param offset_path: Path to the offset file for record random retrieval
        :param mode: A boolean value indicating "train" or "validation" status. Depending on this value, preprocessing
            is done differently.
        :param batch_size: Number of images per batch.
        :param shuffle: Whether or not to shuffle the data in the dataset and the initial file list.
        :param buffer_size: Number of image dirs used as buffer for TensorFlows shuffling of the dataset. 
            If not specified, the entire txt_file will be buffered into memory for shuffling.
        """
        self.csv_file_path = csv_file_path
        self.offset_file_path = offset_path
        self._read_txt_file()
        self.is_training = is_training

        # initial shuffling of the file
        if shuffle:
            np.random.shuffle(self.img_paths)

        self.buffer_size = buffer_size if buffer_size is not None else len(self.img_paths)

        # convert img_paths list to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        # print os.path.abspath(os.path.pardir)
        dataset = data.Dataset.from_tensor_slices(self.img_paths)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        # preprocess img by rescaling, warping, returns bgr image and labels on the fly
        dataset = dataset.map(lambda imgpath: tf.py_func(self._data_augment, [imgpath], [tf.float32, tf.float32]))
        # batch sizing data
        self.data = dataset.batch(batch_size)



    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        with open(self.txt_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.img_paths.append(line.strip())

    def _data_augment(self, imgpath, stdev=0.1):
        """
           This function takes a single imagepath as input, warp it randomly and returns 4 numbers as targets
        """
        f = np.load(DATADIR + os.path.splitext(imgpath)[0] + '.npz')

        im = img_as_float(f[f.keys()[0]])
        target = np.random.normal(scale=stdev, size=4).astype(np.float32)
        warped = warp_image(im, *target)

        # rescale and rgb->bgr, subtract mean. These are pre-processing steps
        warped_scaled_bgr = (warped * 255.0)[:,:,::-1]
        bgr = (warped_scaled_bgr - VGG_MEAN).astype(np.float32)
        return bgr, target
