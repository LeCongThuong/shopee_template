import logging

from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from project.utils import read_csv, get_mean_batch_image_size
from transformers import AutoTokenizer
import torchvision

logger = logging.getLogger(__name__)


class AdaptiveResizerDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform, text_preprocess, tokenizer_str, text_max_length, text_padding='longest',
                 is_truncate=True, do_train=True,  **kwargs):
        self.do_train = do_train
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.text_process = text_preprocess
        self.image_path_list, self.title_list, self.label_group_list = self.set_up(image_dir, csv_file)
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        self.text_max_length = text_max_length
        self.text_padding = text_padding
        self.is_truncate = is_truncate

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        title = self.title_list[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # transform data
        image = self.transform(image=image)
        title = self.text_process(title)

        tokenized = self.tokenizer([title], padding=self.text_padding, truncation=self.is_truncate, max_length=self.text_max_length,
                                   return_tensors="pt")

        if not self.do_train:
            return {"image": image, "title_ids": torch.as_tensor(tokenized["input_ids"][0]),
                    "attention_mask": torch.as_tensor(tokenized["attention_mask"][0])}

        label_group = self.label_group_list[idx]

        return {"images": image, "title_ids": torch.as_tensor(tokenized["input_ids"][0]),
                "attention_masks": torch.as_tensor(tokenized["attention_mask"][0]), "label_groups": torch.as_tensor(label_group)}

    def set_up(self, image_dir, csv_file):
        df = read_csv(csv_file)
        image_path_list = [os.path.join(image_dir, image_name) for image_name in df['image'].tolist()]
        title_list = df['title'].tolist()
        if not self.do_train:
            title_list = list(map(self.text_process, title_list))
            return image_path_list, title_list, None
        return image_path_list, title_list, df['label_group'].to_numpy()

    # def collate_fn(self, batch):
    #     title_ids = torch.stack([x["title_ids"] for x in batch])
    #     attention_masks = torch.stack([x['attention_mask'] for x in batch])
    #     mean_image_size = get_mean_batch_image_size(batch)
    #     resize_op = torchvision.transforms.Resize(size=mean_image_size)
    #     images = torch.stack([resize_op(x['image']) for x in batch])
    #     if not self.do_train:
    #         return {"title_ids": title_ids, "attention_masks": attention_masks, "images": images}
    #     label_groups = torch.stack(x['label_group'] for x in batch)
    #     return {"title_ids": title_ids, "attention_masks": attention_masks, "images": images, "label_groups": label_groups}


class ShopeeDatasetLoader:
    def __init__(self, dataset, sampler, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.sampler = sampler

    def get_dataloader(self):
        return DataLoader(self.dataset, sampler=self.sampler, batch_size=self.batch_size)

    # def get_sampler(self, sampler_str, dataset):
    #     if sampler_str == 'random':
    #         sampler = torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.RandomSampler(dataset),
    #                                                         batch_size=self.batch_size, drop_last=False)
    #     elif sampler_str == 'sequence':
    #         sampler = torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.SequentialSampler(dataset),
    #                                                         batch_size=self.batch_size, drop_last=False)
    #     else:
    #         raise Exception("Type of sampler is not valid")
    #     return sampler

    def show_images(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size)
        res = next(iter(loader))
        images = res['images'].squeeze()

        fig, axes = plt.subplots(nrows=self.batch_size // 4 if self.batch_size % 4 == 0 else self.batch_size // 4 + 1, ncols=4,
                                 figsize=(12, 9))
        for i in range(self.batch_size):
            axes[i//4, i % 4].imshow(np.clip(images[i], 0, 1))




