import os
import argparse
import json
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
from Student.utils import CULane_dataset
from Student.utils.transforms import *
from Student.model_KD import _KDNET
from Student.model import NET
from Teacher.model import _TNET
from calculate import score
import Cdataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default='/home/arc-cjl2902/LaneDetection2/Student/config.json')
    parser.add_argument("--Teacher_exp", type=str, default='/home/arc-cjl2902/LaneDetection2/Teacher/exp')
    parser.add_argument("--KD_exp", type=str, default='/home/arc-cjl2902/LaneDetection2/Student/exp')
    parser.add_argument("--Student_exp", type=str, default='/home/arc-cjl2902/LaneDetection2/Student/exp')
    parser.add_argument("--data_dir", type=str, default='/home/arc-cjl2902/Dataset/50_CULaneDataset')
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--beta", type=float, default=0.3)
    args = parser.parse_args()
    return args


args = parse_args()
config = args.config_dir
with open(config) as f:
    cfg = json.load(f)
print(cfg)
# pars
resize_shape = tuple(cfg['dataset']['resize_shape'])
device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transforms = Resize(resize_shape)
transform = Compose(Rotation(2), ToTensor(), Normalize(mean=mean, std=std))
transform_train = Compose(transforms, transform)
dataset_name = cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(Cdataset, dataset_name)
train_dataset = Dataset_Type(args.data_dir, "train", transform_train)
random_sampler = sampler.RandomSampler(data_source=train_dataset)
batch_sampler = sampler.BatchSampler(random_sampler, 10, drop_last=False)
remain_dataset = DataLoader(train_dataset, sampler=batch_sampler, collate_fn=train_dataset.collate, num_workers=2)
for batch, sample in enumerate(remain_dataset):
    img = sample['image'].to(device0)
    print('------------------------')
print('================')
