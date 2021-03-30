import logging

from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from project.utils import read_csv
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class AdaptiveResizerDataset(Dataset):
    def __init__(self, image_dir, csv_annotation_file, transform, text_preprocess, tokenizer_str, text_max_length, text_padding='longest',
                 is_truncate=True, do_train=True,  **kwargs):
        self.do_train = do_train
        self.image_dir = image_dir
        self.csv_annotation_file = csv_annotation_file
        self.text_process = text_preprocess
        self.image_path_list, self.image_size_list, self.title_list, self.label_group_list = self.set_up(image_dir, csv_annotation_file)
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        self.text_max_length = text_max_length
        self.text_padding = text_padding
        self.is_truncate = is_truncate

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_size_batch = self.image_size_list[idx]
        image_mean_size = np.mean(image_size_batch[:, 1])
        if image_mean_size < 300:
            image_mean_size = 224
        elif 300 <= image_mean_size < 600:
            image_mean_size = 448
        elif 600 <= image_mean_size <= 900:
            image_mean_size = 720
        else:
            image_mean_size = 1024

        image_list = []
        for index in idx:
            image_path = self.image_path_list[index]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_mean_size, image_mean_size))
            if self.transform is not None:
                image = self.transform(image)["image"]
            image_list.append(image)
        np_image_batch = np.stack(image_list, axis=0)

        title_batch = [self.title_list[i] for i in idx]

        tokenized = self.tokenizer(title_batch, padding=self.text_padding, truncation=self.is_truncate, max_length=self.text_max_length,
                                   return_tensors="pt")
        if not self.do_train:
            return {"images": torch.as_tensor(np_image_batch, dtype=torch.float), "title_ids": torch.as_tensor(tokenized["input_ids"]),
                    "attention_masks": torch.as_tensor(tokenized["attention_mask"])}

        label_groups = self.label_group_list[idx]

        return {"images": torch.as_tensor(np_image_batch, dtype=torch.float), "title_ids": torch.as_tensor(tokenized["input_ids"]),
                "attention_masks": torch.as_tensor(tokenized["attention_mask"]), "label_groups": torch.as_tensor(label_groups)}

    def set_up(self, image_dir, csv_file):
        df = read_csv(csv_file)
        image_path_list = [os.path.join(image_dir, image_name) for image_name in df['image'].tolist()]
        title_list = df['title'].tolist()
        if not self.do_train:
            image_size_list = self.get_image_size(image_path_list)
            title_list = list(map(self.text_process, title_list))
            return image_path_list, image_size_list, title_list, image_size_list, None, None
        return image_path_list, np.array(df['image_size'].to_list()), title_list, df['label_group'].to_numpy()

    def get_image_size(self, image_path_list):
        image_size_list = []
        for image_path in image_path_list:
            image = cv2.imread(image_path)
            image_size_list.append(list(image.shape))
        return np.array(image_size_list)


class ShopeeDatasetLoader:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset

    def get_dataloader(self, sampler):
        return DataLoader(self.dataset, sampler=sampler)

    def show_images(self):
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(self.adaptiveResizerDataModule),
            batch_size=self.batch_size,
            drop_last=False)

        loader = DataLoader(self.adaptiveResizerDataModule, sampler=sampler)

        res = next(iter(loader))
        images = res['images'].squeeze()

        fig, axes = plt.subplots(nrows=self.batch_size // 4 if self.batch_size % 4 == 0 else self.batch_size // 4 + 1, ncols=4,
                                 figsize=(12, 9))
        for i in range(self.batch_size):
            axes[i//4, i % 4].imshow(np.clip(images[i], 0, 1))




