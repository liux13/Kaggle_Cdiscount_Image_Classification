import bson
import numpy as np
from googletrans import Translator
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
import os
import io

DATA_DIR = "/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/data/"
translator = Translator()
category_file_path = "/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/data/categories.csv"
categories_df = pd.read_csv(os.path.join(DATA_DIR, "categories.csv"), index_col=0)
train_offsets_df = pd.read_csv(os.path.join(DATA_DIR, "train_offsets.csv"), index_col=0)
VGG_MEAN = np.array([103.939, 116.779, 123.68]).reshape((1,1,3))

def visualize_batch(batch, display_num=2, category_file_path=category_file_path):

    x1, x2, isSame, x1_labels, x2_labels = batch
    batch_size = len(isSame)

    fig, axes = plt.subplots(display_num, 8)
    fig.set_size_inches(2.5 * 8, 2.5 * display_num)

    for row in range(display_num):
        prod_pair_idx = row if display_num >= batch_size else np.random.randint(0, batch_size)
        for col in range(8):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

            if col == 0:
                record = categories_df[categories_df.category_idx == x1_labels[prod_pair_idx]]
                lvl3 = translator.translate(record.category_level3.values[0].lower(), src='fr').text
                axes[row, col].set_title(lvl3)
            elif col == 4:
                record = categories_df[categories_df.category_idx == x2_labels[prod_pair_idx]]
                lvl3 = translator.translate(record.category_level3.values[0].lower(), src='fr').text
                axes[row, col].set_title(lvl3)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            plt.sca(axes[row, col])
            if col < 4:
                pic = (x1[prod_pair_idx, col, :, :, :] +VGG_MEAN)[:,:,::-1]/ 255.
            else:
                pic = (x2[prod_pair_idx, (col-4), :, :, :] + VGG_MEAN)[:,:,::-1]/ 255.
            plt.imshow(pic)

def get_product(product_id):
    record = train_offsets_df[train_offsets_df.index == product_id]
    num_imgs = record.num_imgs
    offset = record.offset.values[0]
    length = record.length.values[0]

    with open(os.path.join(DATA_DIR, "train.bson")) as b:
        b.seek(offset)
        sample = b.read(length)
        item = bson.BSON(sample).decode()

        pic = []
        for i in range(num_imgs):
            pic.append(imread(io.BytesIO(item['imgs'][i]['picture'])))
    return pic

def plot_lvl2_products(lvl2_str):
    category_ids = categories_df[categories_df.category_level2 == lvl2_str].index
    random_category_id = np.random.choice(category_ids)
    lvl3_str = categories_df[categories_df.index == random_category_id].category_level3.values[0]

    product_id = np.random.choice(
        train_offsets_df[train_offsets_df.category_id == random_category_id].index)
    pic = get_product(product_id)

    return pic, translator.translate(lvl3_str.lower(), src='fr').text

def plot_lvl3_products(lvl3_str):
    category_id = categories_df[categories_df.category_level3==lvl3_str].index
    assert len(category_id) == 1

    product_id = np.random.choice(
        train_offsets_df[train_offsets_df.category_id == category_id[0]]
            .index)
    pic = get_product(product_id)
    return pic, translator.translate(lvl3_str.lower(), src='fr').text