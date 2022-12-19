import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CULane(Dataset):
    def __init__(self, path, image_set, transforms=None):
        super(CULane, self).__init__()
        assert image_set in ('train', 'val', 'test'), 'Image_set is not valid'
        self.dir_path = path
        self.image_set = image_set
        self.transforms = transforms

        if image_set != 'test':
            self.creat_index()
        else:
            self.creat_test_index()

    def creat_index(self):
        file_list = os.path.join(self.dir_path, 'list', '{}_gt_do_adam.txt'.format(self.image_set))

        self.img_list = []
        self.seglabel_list = []
        # self.exist_list = []

        with open(file_list) as f:
            for line in f:
                line = line.strip()
                l = line.split(' ')
                self.img_list.append(os.path.join(self.dir_path, l[0][1:]))  # 去除第一个"/"; 图
                self.seglabel_list.append((os.path.join(self.dir_path, l[1][1:])))  # seg_label
                # self.exist_list.append([int(x) for x in l[2:]])  # the lane; 1:exist,0:not exist

    def creat_test_index(self):
        file_list = os.path.join(self.dir_path, 'list', '{}.txt'.format(self.image_set))

        self.img_list = []
        with open(file_list) as f:
            for line in f:
                line = line.strip()
                self.img_list.append(os.path.join(self.dir_path, line[1:]))

    def __getitem__(self, index):
        image = cv2.imread(self.img_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_set != 'test':
            seg_label = cv2.imread(self.seglabel_list[index])[:, :, 0]
            # exist = np.array(self.exist_list[index])
        else:
            seg_label = None
            # exist = None

        sample = {
            'image': image,
            'seg_label': seg_label,
            # 'exist': exist,
            'image_name': self.img_list[index]
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['image'], torch.Tensor):
            # pr = [b['image'] for b in batch]  # 为什么要stack
            img = torch.stack([b['image'] for b in batch])  #
        else:
            img = [b['image'] for b in batch]

        if batch[0]['seg_label'] is None:
            seg_label = None
            # exist = None
        elif isinstance(batch[0]['seg_label'], torch.Tensor):
            seg_label = torch.stack([b['seg_label'] for b in batch])
            # exist = torch.stack([b['exist'] for b in batch])
        else:
            seg_label = [b['seg_label'] for b in batch]
            # exist = [b['exist'] for b in batch]

        sample = {
            'image': img,
            'seg_label': seg_label,
            # 'exist': exist,
            'image_name': [x['image_name'] for x in batch]
        }

        return sample
