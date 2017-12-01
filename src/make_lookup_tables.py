import os
import pandas as pd
from tqdm import *
import struct
import bson
from collections import defaultdict
import numpy as np
from functools import wraps
from googletrans import Translator
from unidecode import unidecode

translator = Translator()

"""
This script is inspired by part 1:
https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
"""

DATA_DIR = "/home/femianjc/CSE627/Kaggle_Cdiscount_Image_Classification/data/"

def memoize(func):
    cache = {}
    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrap

@memoize
def translate_category(x):
    # print x.lower()
    try:
        return unidecode(translator.translate(x.lower(), src='fr').text)
    except:
        return unidecode(x)

def make_offsets_df(bson_path=os.path.join(DATA_DIR, "train.bson"),
                    num_records=7069896,
                    with_categories=True,
                    out_name="train_offsets.csv"):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON(item_data).decode()

            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    df.to_csv(os.path.join(DATA_DIR, out_name))
    print "Offset table saved as: ", os.path.join(DATA_DIR, out_name)
    return df


def make_categories_df(category_names_file=os.path.join(DATA_DIR, "category_names.csv")):
    categories_df = pd.read_csv(category_names_file, index_col=0)
    categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)
    # categories_df[categories_df.columns[0] + "_EN"] = categories_df.category_level1.map(translate_category)
    # categories_df[categories_df.columns[1] + "_EN"] = categories_df.category_level2.map(translate_category)


    path = os.path.join(DATA_DIR, "categories.csv")
    categories_df.to_csv(path, index=True)
    print "Category to idx lookup table saved as: ", path

def make_category_tables(categories_df_path=os.path.join(DATA_DIR, "categories.csv")):
    cat2idx = {}
    idx2cat = {}
    categories_df = pd.read_csv(categories_df_path, index_col=0)
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

def train_val_split(df, category_df_path=os.path.join(DATA_DIR, "categories.csv"), split_percentage=0.2):
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    assert len(df.columns) == 4
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])

    train_list = []
    val_list = []
    cat2idx, idx2cat = make_category_tables(category_df_path)

    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            # Randomly choose the products that become part of the validation set.
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []

            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()

    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)

    train_path = os.path.join(DATA_DIR, "train_images.csv")
    val_path = os.path.join(DATA_DIR, "val_images.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print "Training images saved as: ", train_path
    print "Validation images saved as: ", val_path