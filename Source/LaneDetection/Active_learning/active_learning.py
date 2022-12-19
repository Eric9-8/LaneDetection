import os
import argparse
import json
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from Student.utils import CULane_dataset
from Student.utils.transforms import *
from Student.model_KD import _KDNET
from Student.model import NET
from Teacher.model import _TNET
import Cdataset
from Student.test2 import set_seed, encoder, score, uncertain, div

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default='/home/arc-cjl2902/LaneDetection2/Student/config.json')
    parser.add_argument("--Teacher_exp", type=str, default='/home/arc-cjl2902/LaneDetection2/Teacher/exp/v1')
    parser.add_argument("--KD_exp", type=str, default='/home/arc-cjl2902/LaneDetection2/Student/exp/KDv1')
    parser.add_argument("--Student_exp", type=str, default='/home/arc-cjl2902/LaneDetection2/Student/exp/v1')
    parser.add_argument("--data_dir", type=str, default='/home/arc-cjl2902/Dataset/50_CULaneDataset')
    parser.add_argument("--round", type=int, default=2)

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
transform = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform_train = Compose(transforms, transform)
dataset_name = cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(Cdataset, dataset_name)
train_dataset = Dataset_Type(args.data_dir, "set0", transform_train)

remain_dataset = DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate, num_workers=2)

# Load model
modelS_KD = _KDNET(backbone='regnet_y_400mf', pretrained=False)
modelS = NET(backbone='regnet_y_400mf', pretrained=False)
modelT = _TNET(backbone='regnet_y_3_2gf', pretrained=False)

KD_name = os.path.join(args.KD_exp, 'StudentKD' + '_best.pth')
S_name = os.path.join(args.Student_exp, 'StudentV1' + '_best.pth')
T_name = os.path.join(args.Teacher_exp, 'TeacherV1' + '_best.pth')

KD_dict = torch.load(KD_name, map_location='cpu')
S_dict = torch.load(S_name, map_location='cpu')
T_dict = torch.load(T_name, map_location='cpu')

print('\nloading', KD_name, '------------- From Epoch:', KD_dict['epoch'])
print('\nloading', S_name, '------------- From Epoch:', S_dict['epoch'])
print('\nloading', T_name, '------------- From Epoch:', T_dict['epoch'])

modelS_KD.load_state_dict(KD_dict['net'])
modelS.load_state_dict(S_dict['net'])
modelT.load_state_dict(T_dict['net'])

for name, paras in modelS_KD.named_parameters():
    paras.requires_grad = False
for name, paras in modelS.named_parameters():
    paras.requires_grad = False
for name, paras in modelT.named_parameters():
    paras.requires_grad = False

modelS_KD.to(device0)
modelS.to(device0)
modelT.to(device0)
encoder = encoder(device=device0)

modelS_KD.eval()
modelS.eval()
modelT.eval()
encoder.eval()


def select():
    progress = tqdm(range(len(remain_dataset)))
    with torch.no_grad():
        map_t = []
        map_s = []
        map_kd = []
        map_feature = []
        name = []
        label = []
        for batch, sample in enumerate(remain_dataset):
            img = sample['image'].to(device0)
            img_name = sample['image_name']
            label_name = sample['label_name']

            seg_Teacher = modelT(img)[0]
            seg_KDistillation = modelS_KD(img)[0]
            seg_Student = modelS(img)[0]

            seg_Teacher = F.softmax(seg_Teacher, dim=1)
            map_t.append(seg_Teacher)
            # seg_Teacher = seg_Teacher.detach().cpu().numpy()

            seg_KDistillation = F.softmax(seg_KDistillation, dim=1)
            map_kd.append(seg_KDistillation)
            # seg_KDistillation = seg_KDistillation.detach().cpu().numpy()

            seg_Student = F.softmax(seg_Student, dim=1)
            map_s.append(seg_Student)
            # seg_Student = seg_Student.detach().cpu().numpy()

            feature = encoder(img)
            map_feature.append(feature)

            name.append(img_name[0])
            label.append(label_name[0])
            progress.update(1)
    progress.close()
    un = uncertain(map_kd, map_s, map_t)
    di = div(map_feature)
    name_, label_ = score(un, di, name, label)
    return name_, label_


# def main():
#     name, label = select()

# r = sorted(rec.items(), key=lambda kv: (kv[1], kv[0]))
# save_dir = os.path.join('./', 'record.txt')
#
# with open(save_dir, 'w') as f:
#     for val, key in enumerate(rec):
#         print("{} {}".format(key, val), end="\n", file=f)


if __name__ == '__main__':
    set_seed()
    name, label = select()
    torch.cuda.empty_cache()
    # save_dir = os.path.join('./', 'set_{}_record.txt'.format(0))
    # with open(save_dir, 'w') as f:
    #     for _ in range(len(name)):
    #         print("{} {}".format(name[_][42:], label[_][42:]), end="\n", file=f)
    tst_path = os.path.join(args.data_dir, 'list', 'train_gt_do.txt')
    with open(tst_path, 'a') as f:
        for _ in range(len(name)):
            print("{} {}".format(name[_][42:], label[_][42:]), end="\n", file=f)
