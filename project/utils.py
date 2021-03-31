import pandas as pd
import numpy as np


def read_csv(data_file):
    if data_file.endswith(".pkl"):
        df = pd.read_pickle(data_file)
    else:
        df = pd.read_csv(data_file)
    return df


def get_mean_batch_image_size(tensor_batch):
    image_height_list = []
    for item in tensor_batch:
        image = item['image']
        image_height = image.size()[0]
        image_height_list.append(image_height)
    mean_image_size = np.array(image_height_list).mean()
    return mean_image_size
