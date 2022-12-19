import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class DataTest(Dataset):
    def __init__(self, path, image_set, transform=None):
        self.dir_path = path
        self.image_set = image_set
        self.transform = transform
        self.creat_idx()

    def creat_idx(self):
        file_list = os.path.join(self.dir_path, 'list', '{}_gt_re.txt'.format(self.image_set))

        self.img_list = []
        self.seglabel_list = []

        with open(file_list) as f:
            for line in f:
                line = line.strip()
                l = line.split(' ')
                self.img_list.append(os.path.join(self.dir_path, l[0][1:]))  # 去除第一个"/"; 图
                self.seglabel_list.append((os.path.join(self.dir_path, l[1][1:])))  # seg_label

    def __getitem__(self, index):
        image = cv2.imread(self.img_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seg_label = cv2.imread(self.seglabel_list[index])[:, :, 0]
        sample = {
            'image': image,
            'seg_label': seg_label,
            'image_name': self.img_list[index],
            'label_name': self.seglabel_list[index]
        }
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['image'], torch.Tensor):
            img = torch.stack([b['image'] for b in batch])  #
        else:
            img = [b['image'] for b in batch]

        if batch[0]['seg_label'] is None:
            seg_label = None
        elif isinstance(batch[0]['seg_label'], torch.Tensor):
            seg_label = torch.stack([b['seg_label'] for b in batch])
        else:
            seg_label = [b['seg_label'] for b in batch]

        sample = {
            'image': img,
            'seg_label': seg_label,
            'image_name': [x['image_name'] for x in batch],
            'label_name': [x['label_name'] for x in batch]
        }

        return sample
