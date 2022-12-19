import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Teacher.model import _TNET
from Student.model import NET
from Student.utils.transforms import *
from Student.utils import CULane_dataset
# import CULane_dataset_random
import argparse
import random
import os
import json
from tqdm import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

r = 8


def parse_args(rond):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Student/exp_v3/KDv3_{}'.format(rond))
    parser.add_argument("--config_dir", type=str, default='./Student/config.json')
    parser.add_argument("--data_dir", type=str, default='/home/arc-cjl2902/Dataset/50_CULaneDataset')
    args = parser.parse_args()
    return args


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed()
args = parse_args(r)
exp_dir = args.exp_dir
exp_name = exp_dir.split('/')[-1]

config = args.config_dir
with open(config) as f:
    cfg = json.load(f)
resize_shape = tuple(cfg['dataset']['resize_shape'])
device = torch.device('cuda')

__all__ = ['SegmentationMetric']


class SegmentationMetric(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusionMatrix = np.zeros((self.num_class,) * 2)

    def pixel_accuracy(self):
        """
        :return: acc = (TP+TN)/(TP+TN+FP+FN)
        """
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        acc = round(acc, 5)
        return acc

    def class_pixel_accuracy(self):
        """
        :return:acc = (TP) / (TP+FP)
        """
        class_acc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return class_acc

    def mean_pixel_accuracy(self):
        class_acc = self.class_pixel_accuracy()
        mean_acc = np.nanmean(class_acc)
        return mean_acc

    def precision(self):
        precis = np.diag(self.confusionMatrix)[0] / self.confusionMatrix.sum(axis=0)[0]
        return precis

    def recall(self):
        rec = np.diag(self.confusionMatrix)[0] / self.confusionMatrix.sum(axis=1)[0]
        return rec

    def f1_score(self):
        f_score = (2 * self.precision() * self.recall()) / (self.precision() + self.recall())
        return f_score

    def mean_intersection_over_union(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        mIoU = round(mIoU, 4)
        return IoU[1], mIoU

    def gen_confusion_matrix(self, pre, gt):
        mask = (gt >= 0) & (gt < self.num_class)
        label = self.num_class * gt[mask] + pre[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def frequency_weight_intersection_over_union(self):
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def add_batch(self, pre, gt):
        assert pre.shape == gt.shape
        self.confusionMatrix += self.gen_confusion_matrix(pre, gt)
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.num_class, self.num_class))


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
dataset_name = cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(CULane_dataset, dataset_name)
transform_test_img = Resize(resize_shape)
transforms = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform = Compose(transform_test_img, transforms)
test_dataset = Dataset_Type(args.data_dir, "val", transform)
test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=test_dataset.collate, num_workers=4,
                         worker_init_fn=worker_init_fn)

net = NET(backbone='regnet_y_400mf', pretrained=False)
# net = _TNET(backbone='regnet_y_3_2gf', pretrained=False)
#
# save_name = os.path.join(exp_dir, 'StudentV3_{}'.format(r) + '_best.pth')
# save_name = os.path.join(exp_dir, 'TeacherV3_{}'.format(r) + '_best.pth')
save_name = os.path.join(exp_dir, 'StudentKDV3_{}'.format(r) + '_best.pth')
save_dict = torch.load(save_name, map_location='cpu')
print('\nloading', save_name, '------------- From Epoch:', save_dict['epoch'])
net.load_state_dict(save_dict['net'])
for name, paras in net.named_parameters():
    paras.requires_grad = False
net.eval().to(device)

acc_sum = 0
mIoU_sum = 0
IoU_sum = []
f1_sum = 0
cc = len(test_loader)
progress = tqdm(range(len(test_loader)))
with torch.no_grad():
    for batch, sample in enumerate(test_loader):
        img = sample['image'].to(device)
        seg_label = sample['seg_label'].to(device)
        seg_label = torch.where(seg_label == 2, 1, seg_label)
        seg_label = torch.where(seg_label == 3, 1, seg_label)
        seg_label = torch.where(seg_label == 4, 1, seg_label)
        img_name = sample['image_name']
        # exist = sample['exist'].to(device)
        seg_pre = net(img)[0]
        # exist_pre = net(img)[1]
        # exist_pre = F.softmax(exist_pre)
        #
        # exist_pre = exist_pre.detach().cpu().numpy()

        seg_pre = F.softmax(seg_pre, dim=1)
        seg_pre = seg_pre.detach().cpu().numpy()
        seg_label = seg_label.detach().cpu().numpy()

        for b in range(len(seg_pre)):
            seg_ = seg_pre[b]
            label = seg_label[b]
            image_name = img_name[b]

            seg = np.ascontiguousarray(np.transpose(seg_, (1, 2, 0)))
            prc = seg.argmax(axis=-1)

            img_pre = prc
            img_label = label

            metrix = SegmentationMetric(2)
            metrix.add_batch(img_pre, img_label)
            acc = metrix.pixel_accuracy()
            acc_sum += acc
            IoU, mIoU = metrix.mean_intersection_over_union()
            IoU_sum.append(IoU)
            mIoU_sum += mIoU
            f1 = metrix.f1_score()
            f1_sum += f1
        progress.update(1)
progress.close()

# print("%s.jpg", split_path(img_name[b])[-1])
# print("accuracy: " + str(acc * 100) + "%")
# print("mIoU: " + str(mIoU))
# print("IoU: " + str(round(IoU, 4)))
# print("f1: " + str(f1))
# print("-----------------------------")


acc_sum = acc_sum / len(test_loader)
mIoU_sum = mIoU_sum / len(test_loader)
IoU_sum = np.nanmean(IoU_sum)
f1_sum = f1_sum / len(test_loader)
acc_sum = round(acc_sum, 5)
mIoU_sum = round(mIoU_sum, 4)
IoU_sum = round(IoU_sum, 4)
f1_sum = round(f1_sum, 3)

print("All accuracy: " + str((acc_sum * 100) / 4) + "%")
print("All mIoU: " + str((mIoU_sum) / 4))
print("All Iou: " + str(IoU_sum))
print("All f1: " + str(f1))
