"""
1.训练三个base模型model_t,model_S,model_kd（已完成）
2.for i in range(44):
     set{i} --> active_learning (model_t{i-1},model_s{i-1},model_kd{i-1})
    -->Training (model_t{i},model_s{i},model_kd{i})
"""
import argparse
import json
import shutil
import random
import os
import time
import math
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
# from torch.optim import lr_scheduler
# from Student.utils import CULane_dataset, lr_scheduler
from Student.utils import CULane_dataset
import lr_scheduler
from Student.utils.transforms import *
from Student.model import NET
from Student.model_KD import _KDNET
from Teacher.model import _TNET
from Active_learning import Cdataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'


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


def parse_args(r):
    """
    :param r: select round
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--config_dir", type=str, default='/home/arc-cjl2902/LaneDetection2/Student/config.json')
    parser.add_argument("--data_dir", type=str, default='/home/arc-cjl2902/Dataset/50_CULaneDataset')
    parser.add_argument("--t_weight", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Teacher/exp_v3/v3_{}'.format(r))
    parser.add_argument("--s_weight", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Student/exp_v3/v3_{}'.format(r))
    parser.add_argument("--kd_weight", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Student/exp_v3/KDv3_{}'.format(r))
    parser.add_argument("--logdir_t", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Teacher/new_log_v3/TeacherV3_{}'.format(r))
    parser.add_argument("--logdir_s", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Student/new_log_v3/StudentV3_{}'.format(r))
    parser.add_argument("--logdir_kd", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Student/new_log_v3/KDV3_{}'.format(r))
    parser.add_argument("--round", type=int, default=r)
    args = parser.parse_args()
    return args


def datasets(args, cfg):
    resize_shape = tuple(cfg['dataset']['resize_shape'])
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # transform_train = Compose(Resize(resize_shape), Rotation(2), addNoise(10), HFlip(0.2), VFlip(0.2), ToTensor(),
    #                           Normalize(mean=mean, std=std))
    # transform_train = Compose(Resize(resize_shape), Rotation(2), ToTensor(), Normalize(mean=mean, std=std))
    transform_train = Compose(Resize(resize_shape), Rotation(2), HFlip(0.2), VFlip(0.2), ToTensor(),
                              Normalize(mean=mean, std=std))
    dataset_name = 'CULane'
    Dataset_Type = getattr(CULane_dataset, dataset_name)
    train_dataset = Dataset_Type(args.data_dir, "train", transform_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg['dataset']['batch_size'], shuffle=True,
                              collate_fn=train_dataset.collate, num_workers=6, worker_init_fn=worker_init_fn)
    transform_val_img = Resize(resize_shape)
    transform_val_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
    transform_val = Compose(transform_val_img, transform_val_x)

    valid_dataset = Dataset_Type(args.data_dir, 'val', transform_val)
    valid_loader = DataLoader(valid_dataset, batch_size=4, collate_fn=valid_dataset.collate, num_workers=4)
    return train_loader, valid_loader


def select_set(args, cfg):
    resize_shape = tuple(cfg['dataset']['resize_shape'])
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform_train = Compose(Resize(resize_shape), ToTensor(), Normalize(mean=mean, std=std))
    dataset_name = "CULane"
    Dataset_Type = getattr(Cdataset, dataset_name)
    train_dataset = Dataset_Type(args.data_dir, "set{}".format(args.round), transform_train)
    remain_dataset = DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate, num_workers=4)
    print('load selected set{}'.format(args.round))
    return remain_dataset


# def random_set(args):
#     file_list = os.path.join(args.data_dir, 'list', 'set{}_gt_re.txt'.format(args.round + 1))
#     with open(file_list) as f:
#         lines = f.readlines()
#     index = list(range(len(lines)))
#     random.shuffle(index)
#     tst_path = os.path.join(args.data_dir, 'list', 'train_gt_do_random.txt')
#     for i in range(len(index) // 2):
#         with open(tst_path, 'a') as k:
#             print("{}".format(lines[index[i]]), end="\n", file=k)


class Encoder(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True):
        super(Encoder, self).__init__()
        backbone = models.regnet.__dict__[backbone](pretrained=pretrained)
        self.base_layers = list(backbone.children())
        self.layer0 = self.base_layers[0]
        self.layer1 = self.base_layers[1].block1
        self.layer2 = self.base_layers[1].block2
        self.layer3 = self.base_layers[1].block3
        self.layer4 = self.base_layers[1].block4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4


def encoder(args, device):
    set_seed()
    model = Encoder('regnet_y_400mf', pretrained=False)
    save_name = os.path.join(args.s_weight + '/StudentV3_{}'.format(args.round) + '_best.pth')
    save_dict = torch.load(save_name, map_location='cpu')
    model.load_state_dict(save_dict['net'], strict=False)
    for name, paras in model.named_parameters():
        paras.requires_grad = False
    model.to(device)
    model.eval()
    return model


def div(feature_map):
    k = 4
    B = len(feature_map)
    DISTANCE = torch.zeros((B, B))
    count = torch.zeros(B)
    for i in range(B):
        xi = feature_map[i]
        for j in range(i + 1, B):
            xj = feature_map[j]
            DISTANCE[i][j] = torch.dist(xi, xj, p=2)
    lower = np.tril_indices(B, -1)
    DISTANCE[lower] = DISTANCE.T[lower]

    Sorted, idx = torch.sort(DISTANCE)
    DIS_K = Sorted[:, k]

    for m in range(B):
        count[m] = torch.numel(DISTANCE[m][DISTANCE[m] < DIS_K])

    return count


def uncertain(pre_kd, pre_s, pre_t):
    N = len(pre_kd)
    uncertainty = torch.zeros(N)
    for i in range(N):
        ss = torch.dist(pre_kd[i][..., 1], pre_s[i][..., 1], p=2)
        st = torch.dist(pre_kd[i][..., 1], pre_t[i][..., 1], p=2)
        uncertainty[i] = (ss + st) * max(st / ss, ss / st)
    return uncertainty


def score(score1, score2, name, label):
    name_ = []
    label_ = []
    res = score1 + 5 * score2
    _, index = res.topk(500)
    for i in index:
        name_.append(name[i])
        label_.append(label[i])
    return name_, label_


def active_learning(args, remain_dataset, encoder, device):
    modelS_KD = _KDNET(backbone='regnet_y_400mf', pretrained=False)
    modelS = NET(backbone='regnet_y_400mf', pretrained=False)
    modelT = _TNET(backbone='regnet_y_3_2gf', pretrained=False)

    KD_name = os.path.join(args.kd_weight, 'StudentKDV3_{}'.format(args.round) + '_best.pth')
    S_name = os.path.join(args.s_weight, 'StudentV3_{}'.format(args.round) + '_best.pth')
    T_name = os.path.join(args.t_weight, 'TeacherV3_{}'.format(args.round) + '_best.pth')

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

    modelS_KD.to(device)
    modelS.to(device)
    modelT.to(device)

    modelS_KD.eval()
    modelS.eval()
    modelT.eval()
    encoder.eval()

    progress = tqdm(range(len(remain_dataset)))
    with torch.no_grad():
        map_t = []
        map_s = []
        map_kd = []
        map_feature = []
        name = []
        label = []
        for batch, sample in enumerate(remain_dataset):
            img = sample['image'].to(device)
            img_name = sample['image_name']
            label_name = sample['label_name']

            seg_Teacher = modelT(img)[0]
            seg_KDistillation = modelS_KD(img)[0]
            seg_Student = modelS(img)[0]

            seg_Teacher = F.softmax(seg_Teacher, dim=1)
            map_t.append(seg_Teacher)

            seg_KDistillation = F.softmax(seg_KDistillation, dim=1)
            map_kd.append(seg_KDistillation)

            seg_Student = F.softmax(seg_Student, dim=1)
            map_s.append(seg_Student)

            feature = encoder(img)
            map_feature.append(feature)

            name.append(img_name[0])
            label.append(label_name[0])
            progress.update(1)
    progress.close()
    un = uncertain(map_kd, map_s, map_t)
    di = div(map_feature)
    name_, label_ = score(un, di, name, label)

    tst_path = os.path.join(args.data_dir, 'list', 'train_gt_do_adam.txt')
    with open(tst_path, 'a') as f:
        for _ in range(len(name_)):
            print("{} {}".format(name_[_][42:], label_[_][42:]), end="\n", file=f)


class STUDENT:
    def __init__(self, device, args, cfg, train_dataset, val_dataset, transform_val_img, record, model, optimizer, lr,
                 best_val_loss, scaler):

        self.device = device
        self.cfg = cfg
        self.args = args
        self.train_loader = train_dataset
        self.valid_loader = val_dataset
        self.transform_val_img = transform_val_img
        self.record = record

        self.model_s = model
        self.optimizer = optimizer
        self.lr_ = lr
        self.best_val_loss = best_val_loss
        self.scaler = scaler

    def train(self, epoch):
        print("Train student epoch is {}".format(epoch))
        self.model_s.train()
        train_loss = 0
        progress = tqdm(range(len(self.train_loader)))
        for batch, sample in enumerate(self.train_loader):
            img = sample['image'].to(self.device)
            seg_label = sample['seg_label'].to(self.device)
            seg_label = torch.where(seg_label == 2, 1, seg_label)
            seg_label = torch.where(seg_label == 3, 1, seg_label)
            seg_label = torch.where(seg_label == 4, 1, seg_label)
            self.optimizer.zero_grad()
            with autocast():
                seg_pre, loss = self.model_s(img, seg_label)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_.step()
            iter_idx = epoch * len(self.train_loader) + batch
            train_loss = loss.item()
            progress.set_description("Student model{} training batch loss {:.3f}:".format(self.args.round, loss.item()))
            progress.update(1)

            lr = self.optimizer.param_groups[0]['lr']
            self.record.add_scalar('/train/train_loss', train_loss, iter_idx)
            self.record.add_scalar('learning_rate', lr, iter_idx)

        progress.close()
        self.record.flush()
        if epoch % 1 == 0:
            save_dict = {
                "epoch": epoch,
                "net": self.model_s.state_dict(),
                "optim": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_.state_dict(),
                "best_val_loss": self.best_val_loss
            }
            save_name = os.path.join(self.args.s_weight + '/StudentV3_{}'.format(self.args.round) + '.pth')
            torch.save(save_dict, save_name)
            print("student model is saved: {}".format(save_name))

        print("==============================\n")

    def valid(self, epoch):
        print("Valid student epoch is {}".format(epoch))

        self.model_s.eval()
        val_loss = 0
        progress = tqdm(range(len(self.valid_loader)))

        with torch.no_grad():
            for batch, sample in enumerate(self.valid_loader):
                img = sample['image'].to(self.device)
                seg_label = sample['seg_label'].to(self.device)
                seg_label = torch.where(seg_label == 2, 1, seg_label)
                seg_label = torch.where(seg_label == 3, 1, seg_label)
                seg_label = torch.where(seg_label == 4, 1, seg_label)

                seg_pre, loss = self.model_s(img, seg_label)

                gap_num = 5
                if batch % gap_num == 0 and batch < 50 * gap_num:
                    origin_img = []
                    seg_pre = seg_pre.detach().cpu().numpy()
                    for b in range(len(img)):
                        img_name = sample['image_name'][b]
                        img = cv2.imread(img_name)
                        img = self.transform_val_img({'image': img})['image']

                        lane_img = np.zeros_like(img)
                        color = np.array([255, 125, 0])
                        color_mask = np.argmax(seg_pre[b], axis=0)

                        lane_img[color_mask == 1] = color[0]

                        img = cv2.addWeighted(src1=lane_img, alpha=0.7, src2=img, beta=1., gamma=0.)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)

                        origin_img.append(img)
                        origin_img.append(lane_img)
                    self.record.add_images("img{}".format(batch), torch.tensor(np.array(origin_img)), epoch,
                                           dataformats='NHWC')

                val_loss += loss.item()
                progress.set_description("Student model{} valid batch loss {:.3f}".format(self.args.round, loss.item()))
                progress.update(1)

        progress.close()

        iter_idx = (epoch + 1) * len(self.train_loader) + batch
        self.record.add_scalar('valid/valid_loss', val_loss, iter_idx)
        self.record.flush()

        print("==============================\n")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print("best_val_loss", self.best_val_loss)
            save_name = os.path.join(self.args.s_weight + '/StudentV3_{}'.format(self.args.round) + '.pth')
            copy_name = os.path.join(self.args.s_weight + '/StudentV3_{}'.format(self.args.round) + '_best.pth')
            shutil.copyfile(save_name, copy_name)


class TEACHER:
    def __init__(self, device, args, cfg, train_dataset, val_dataset, transform_val_img, record, model,
                 optimizer, lr, best_val_loss, scaler):

        self.device = device
        self.cfg = cfg
        self.args = args
        self.train_loader = train_dataset
        self.valid_loader = val_dataset
        self.transform_val_img = transform_val_img
        self.record = record
        self.model_t = model
        self.optimizer = optimizer
        self.lr_ = lr
        self.best_val_loss = best_val_loss
        self.scaler = scaler

    def train(self, epoch):
        print("Train teacher epoch is {}".format(epoch))
        self.model_t.train()
        train_loss = 0
        progress = tqdm(range(len(self.train_loader)))
        for batch, sample in enumerate(self.train_loader):
            img = sample['image'].to(self.device)
            seg_label = sample['seg_label'].to(self.device)
            seg_label = torch.where(seg_label == 2, 1, seg_label)
            seg_label = torch.where(seg_label == 3, 1, seg_label)
            seg_label = torch.where(seg_label == 4, 1, seg_label)

            self.optimizer.zero_grad()

            with autocast():
                seg_pre, loss = self.model_t(img, seg_label)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_.step()
            iter_idx = epoch * len(self.train_loader) + batch
            train_loss = loss.item()
            progress.set_description("Teacher model{} training batch loss {:.3f}:".format(self.args.round, loss.item()))
            progress.update(1)

            lr = self.optimizer.param_groups[0]['lr']
            self.record.add_scalar('/train/train_loss', train_loss, iter_idx)
            self.record.add_scalar('learning_rate', lr, iter_idx)

        progress.close()
        self.record.flush()
        if epoch % 1 == 0:
            save_dict = {
                "epoch": epoch,
                "net": self.model_t.state_dict(),
                "optim": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_.state_dict(),
                "best_val_loss": self.best_val_loss
            }
            save_name = os.path.join(self.args.t_weight + '/TeacherV3_{}'.format(self.args.round) + '.pth')
            torch.save(save_dict, save_name)
            print("teacher model is saved: {}".format(save_name))

        print("==============================\n")

    def valid(self, epoch):
        print("Valid teacher epoch is {}".format(epoch))

        self.model_t.eval()
        val_loss = 0
        progress = tqdm(range(len(self.valid_loader)))

        with torch.no_grad():
            for batch, sample in enumerate(self.valid_loader):
                img = sample['image'].to(self.device)
                seg_label = sample['seg_label'].to(self.device)
                seg_label = torch.where(seg_label == 2, 1, seg_label)
                seg_label = torch.where(seg_label == 3, 1, seg_label)
                seg_label = torch.where(seg_label == 4, 1, seg_label)

                seg_pre, loss = self.model_t(img, seg_label)

                gap_num = 5
                if batch % gap_num == 0 and batch < 50 * gap_num:
                    origin_img = []
                    seg_pre = seg_pre.detach().cpu().numpy()
                    for b in range(len(img)):
                        img_name = sample['image_name'][b]
                        img = cv2.imread(img_name)
                        img = self.transform_val_img({'image': img})['image']

                        lane_img = np.zeros_like(img)
                        color = np.array([255, 125, 0])
                        color_mask = np.argmax(seg_pre[b], axis=0)

                        lane_img[color_mask == 1] = color[0]

                        img = cv2.addWeighted(src1=lane_img, alpha=0.7, src2=img, beta=1., gamma=0.)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)

                        origin_img.append(img)
                        origin_img.append(lane_img)
                    self.record.add_images("img{}".format(batch), torch.tensor(np.array(origin_img)), epoch,
                                           dataformats='NHWC')

                val_loss += loss.item()
                progress.set_description("Teacher model{} valid batch loss {:.3f}".format(self.args.round, loss.item()))
                progress.update(1)

        progress.close()

        iter_idx = (epoch + 1) * len(self.train_loader) + batch
        self.record.add_scalar('valid/valid_loss', val_loss, iter_idx)
        self.record.flush()

        print("==============================\n")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print("best_val_loss", self.best_val_loss)
            save_name = os.path.join(self.args.t_weight + '/TeacherV3_{}'.format(self.args.round) + '.pth')
            copy_name = os.path.join(self.args.t_weight + '/TeacherV3_{}'.format(self.args.round) + '_best.pth')
            shutil.copyfile(save_name, copy_name)


class DISTILLATION:
    def __init__(self, device, args, cfg, train_dataset, val_dataset, transform_val_img, record, model_t,
                 model_kd, optimizer, lr, best_val_loss, scaler):
        self.device = device
        self.cfg = cfg
        self.args = args
        self.train_loader = train_dataset
        self.valid_loader = val_dataset
        self.transform_val_img = transform_val_img
        self.record = record
        self.model_t = model_t
        self.model_kd = model_kd
        self.optimizer = optimizer
        self.lr_ = lr
        self.best_val_loss = best_val_loss
        self.scaler = scaler

    def train(self, epoch):
        print("Train KD epoch is {}".format(epoch))
        self.model_kd.train()
        train_loss = 0
        progress = tqdm(range(len(self.train_loader)))

        for batch, sample in enumerate(self.train_loader):
            img = sample['image'].to(self.device)
            seg_label = sample['seg_label'].to(self.device)
            seg_label = torch.where(seg_label == 2, 1, seg_label)
            seg_label = torch.where(seg_label == 3, 1, seg_label)
            seg_label = torch.where(seg_label == 4, 1, seg_label)

            seg_T = self.model_t(img)[0]
            seg_kd_label = F.softmax(seg_T, dim=1)

            self.optimizer.zero_grad()
            with autocast():
                seg_pre, loss = self.model_kd(img, seg_label, seg_kd_label)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_.step()

            iter_idx = epoch * len(self.train_loader) + batch
            train_loss = loss.item()
            progress.set_description("batch loss {:.3f}:".format(loss.item()))
            progress.update(1)

            lr = self.optimizer.param_groups[0]['lr']
            self.record.add_scalar('/train/train_loss', train_loss, iter_idx)
            self.record.add_scalar('learning_rate', lr, iter_idx)

        progress.close()
        self.record.flush()
        if epoch % 1 == 0:
            save_dict = {
                "epoch": epoch,
                "net": self.model_kd.state_dict(),
                "optim": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_.state_dict(),
                "best_val_loss": self.best_val_loss
            }
            save_name = os.path.join(self.args.kd_weight + '/StudentKDV3_{}'.format(self.args.round) + '.pth')
            torch.save(save_dict, save_name)
            print("KD model is saved: {}".format(save_name))

        print("==============================\n")

    def valid(self, epoch):
        print("Valid KD epoch is {}".format(epoch))

        self.model_kd.eval()
        val_loss = 0
        progress = tqdm(range(len(self.valid_loader)))

        with torch.no_grad():
            for batch, sample in enumerate(self.valid_loader):
                img = sample['image'].to(self.device)
                seg_label = sample['seg_label'].to(self.device)
                seg_label = torch.where(seg_label == 2, 1, seg_label)
                seg_label = torch.where(seg_label == 3, 1, seg_label)
                seg_label = torch.where(seg_label == 4, 1, seg_label)

                seg_T = self.model_t(img)[0]

                seg_kd_label = F.softmax(seg_T, dim=1)

                seg_pre, loss = self.model_kd(img, seg_label, seg_kd_label)

                gap_num = 5
                if batch % gap_num == 0 and batch < 50 * gap_num:
                    origin_img = []
                    seg_pre = seg_pre.detach().cpu().numpy()
                    # exist_pre = exist_pre.detach().cpu().numpy()

                    for b in range(len(img)):
                        img_name = sample['image_name'][b]
                        img = cv2.imread(img_name)
                        img = self.transform_val_img({'image': img})['image']

                        lane_img = np.zeros_like(img)

                        color = np.array([255, 125, 0])
                        color_mask = np.argmax(seg_pre[b], axis=0)
                        lane_img[color_mask == 1] = color[0]

                        img = cv2.addWeighted(src1=lane_img, alpha=0.7, src2=img, beta=1., gamma=0.)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)

                        origin_img.append(img)
                        origin_img.append(lane_img)
                    self.record.add_images("img{}".format(batch), torch.tensor(np.array(origin_img)), epoch,
                                           dataformats='NHWC')

                val_loss += loss.item()
                progress.set_description("batch loss {:.3f}".format(loss.item()))
                progress.update(1)

        progress.close()

        iter_idx = (epoch + 1) * len(self.train_loader) + batch
        self.record.add_scalar('valid/valid_loss', val_loss, iter_idx)
        self.record.flush()

        print("==============================\n")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print("best_val_loss", self.best_val_loss)
            save_name = os.path.join(self.args.kd_weight + '/StudentKDV3_{}'.format(self.args.round) + '.pth')
            copy_name = os.path.join(self.args.kd_weight + '/StudentKDV3_{}'.format(self.args.round) + '_best.pth')
            shutil.copyfile(save_name, copy_name)


def do_s(cfg, args, student):
    for epoch in range(0, cfg['MAX_EPOCHS']):
        student.train(epoch)
        if epoch % 1 == 0:
            print("\nValidation for experiment:", args.s_weight)
            print(time.strftime('%H:%M:%S', time.localtime()))
            student.valid(epoch)
    print('Student model round {} has been trained'.format(args.round))


def do_t(cfg, args, teacher):
    for epoch in range(0, cfg['MAX_EPOCHS']):
        teacher.train(epoch)
        if epoch % 1 == 0:
            print("\nValidation for experiment:", args.t_weight)
            print(time.strftime('%H:%M:%S', time.localtime()))
            teacher.valid(epoch)
    print('Teacher model round {} has been trained'.format(args.round))


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant(m.bias, 0)


def main():
    set_seed()
    device0 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = '/home/arc-cjl2902/LaneDetection2/Student/config.json'
    with open(config) as f:
        cfg = json.load(f)
    print(cfg)
    transforms_val = Resize(tuple(cfg['dataset']['resize_shape']))
    """
    Start Training
    """
    for i in range(9, 44):
        args = parse_args(i)
    #     args = parse_args(96)
        # print('\n-------------Start {}-------------\n'.format(i))
        if not os.path.exists(args.t_weight) or os.path.exists(args.s_weight) or os.path.exists(args.kd_weight):
            os.mkdir(args.t_weight)
            os.mkdir(args.s_weight)
            os.mkdir(args.kd_weight)

        model_t = _TNET(backbone='regnet_y_3_2gf', pretrained=True)
        model_t.to(device0)
        model_s = NET(backbone='regnet_y_400mf', pretrained=True)
        model_s.to(device1)
        train_loader, valid_loader = datasets(args, cfg)

        base_lr = 1e-3
        min_lrs = 1e-8
        total = 10 * len(train_loader)
        warmup = int(total * 0.1)
        max_iter = int(total * 0.8)
        """
        Student
        """
        record_s = SummaryWriter(args.logdir_s)
        # optimizer_s = optim.AdamW(model_s.parameters(), lr=base_lr, weight_decay=1e-6)
        optimizer_s = optim.Adam(model_s.parameters(), lr=base_lr)
        # lambda0 = lambda cur_iter: cur_iter / warmup if cur_iter < warmup else (min_lrs + 0.5 * (base_lr - min_lrs) * (
        #         1.0 * math.cos((cur_iter - warmup) / (max_iter - warmup) * math.pi))) / 0.1
        # lr_s = lr_scheduler.LambdaLR(optimizer_s, lr_lambda=lambda0)
        lr_s = lr_scheduler.PolyLR(optimizer_s, 0.8, max_iter=max_iter, min_lrs=min_lrs, warmup=warmup)
        # lr_s = lr_scheduler.ReduceLROnPlateau(optimizer_s, mode='min', factor=0.1, patience=500, verbose=True, min_lr=1e-8,cooldown=500)
        best_val_loss_s = 1e6
        scaler_s = GradScaler()
        student = STUDENT(device1, args, cfg, train_loader, valid_loader, transforms_val, record_s, model_s,
                          optimizer_s, lr_s, best_val_loss_s, scaler_s)

        # for epoch in range(0, cfg['MAX_EPOCHS']):
        #     student.train(epoch)
        #     if epoch % 1 == 0:
        #         print("\nValidation for experiment:", args.s_weight)
        #         print(time.strftime('%H:%M:%S', time.localtime()))
        #         student.valid(epoch)
        #
        """
        Teacher
        """
        record_t = SummaryWriter(args.logdir_t)
        # optimizer_t = optim.AdamW(model_t.parameters(), lr=base_lr, weight_decay=1e-4)
        optimizer_t = optim.Adam(model_t.parameters(), base_lr)
        lr_t = lr_scheduler.PolyLR(optimizer_t, 0.8, max_iter=max_iter, min_lrs=min_lrs, warmup=warmup)
        best_val_loss_t = 1e6
        scaler_t = GradScaler()
        teacher = TEACHER(device0, args, cfg, train_loader, valid_loader, transforms_val, record_t, model_t,
                          optimizer_t, lr_t, best_val_loss_t, scaler_t)
        #
        # save_dict = torch.load(os.path.join(args.t_weight + '/TeacherV3_{}'.format(args.round) + '.pth'))
        # model_t.load_state_dict(save_dict['net'])
        # optimizer_t.load_state_dict(save_dict['optim'])
        # lr_t.load_state_dict(save_dict['lr_scheduler'])
        # start_epoch = save_dict['epoch'] + 1
        # best_val_loss_t = save_dict['best_val_loss']
        # for epoch in range(start_epoch, cfg['MAX_EPOCHS']):
        #     teacher.train(epoch)
        #     if epoch % 1 == 0:
        #         print("\nValidation for experiment:", args.t_weight)
        #         print(time.strftime('%H:%M:%S', time.localtime()))
        #         teacher.valid(epoch)
        """
        双线程
        """
        task1 = Thread(target=do_s, args=(cfg, args, student))
        task2 = Thread(target=do_t, args=(cfg, args, teacher))
        task1.start()
        task2.start()
        task1.join()
        task2.join()

        """
        KD
        """
        print('-------------Initial KD model--------------------')
        record_kd = SummaryWriter(args.logdir_kd)
        model_T = _TNET(backbone='regnet_y_3_2gf', pretrained=False)

        save_name = os.path.join(args.t_weight, 'TeacherV3_{}'.format(args.round) + '_best.pth')
        save_dict = torch.load(save_name, map_location='cpu')
        print('\nloading', save_name, '------------- From Teacher model Epoch:', save_dict['epoch'])
        model_T.load_state_dict(save_dict['net'])
        for name, paras in model_T.named_parameters():
            paras.requires_grad = False

        model_T.to(device0)
        model_T.eval()

        model_kd = _KDNET(backbone='regnet_y_400mf', pretrained=True)
        model_kd.to(device0)

        # optimizer_kd = optim.AdamW(model_kd.parameters(), lr=base_lr, weight_decay=1e-4)
        optimizer_kd = optim.Adam(model_kd.parameters(), base_lr)
        lr_kd = lr_scheduler.PolyLR(optimizer_kd, 0.8, max_iter=max_iter, min_lrs=min_lrs, warmup=warmup)

        best_val_loss_kd = 1e6
        scaler_kd = GradScaler()
        distillation = DISTILLATION(device0, args, cfg, train_loader, valid_loader, transforms_val, record_kd, model_T,
                                    model_kd, optimizer_kd, lr_kd, best_val_loss_kd, scaler_kd)

        for epoch in range(0, cfg['MAX_EPOCHS']):
            distillation.train(epoch)
            if epoch % 1 == 0:
                print("\nValidation for experiment:", args.kd_weight)
                print(time.strftime('%H:%M:%S', time.localtime()))
                distillation.valid(epoch)
        print('KD model round {} has been trained'.format(args.round))

        """
        Active learning
        """
        torch.cuda.empty_cache()
        remain_dataset = select_set(args, cfg)
        code = encoder(args, device0)
        print('Active learning in process')
        active_learning(args, remain_dataset, code, device0)
        print('Round {} has been completed, {} rounds remaining'.format(args.round, 43 - args.round))

        """
        Random
        torch.cuda.empty_cache()
        """


if __name__ == '__main__':
    main()
