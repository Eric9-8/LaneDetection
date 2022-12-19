import argparse
import json
import shutil
import random
import os
import time
from threading import Thread
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
# from Student.utils import CULane_dataset, lr_scheduler
import CULane_dataset_random
import lr_scheduler
from Student.utils.transforms import *
from Student.model import NET

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args(r):
    """
    :param r: select round
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--config_dir", type=str, default='/home/arc-cjl2902/LaneDetection2/Student/config.json')
    parser.add_argument("--data_dir", type=str, default='/home/arc-cjl2902/Dataset/50_CULaneDataset')
    parser.add_argument("--t_weight", type=str, default='/home/arc-cjl2902/LaneDetection2/Teacher/exp/v1_{}'.format(r))
    parser.add_argument("--s_weight", type=str, default='/home/arc-cjl2902/LaneDetection2/Student/exp/v1_{}'.format(r))
    parser.add_argument("--kd_weight", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Student/exp/KDv1_{}'.format(r))
    parser.add_argument("--logdir_t", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Teacher/new_log/TeacherV1_{}'.format(r))
    parser.add_argument("--logdir_s", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Student/new_log/StudentV1_{}'.format(r))
    parser.add_argument("--logdir_kd", type=str,
                        default='/home/arc-cjl2902/LaneDetection2/Student/new_log/KDV1_{}'.format(r))
    parser.add_argument("--round", type=int, default=r)
    args = parser.parse_args()
    return args


def datasets(args, cfg):
    resize_shape = tuple(cfg['dataset']['resize_shape'])
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform_train = Compose(Resize(resize_shape), Rotation(2), ToTensor(), Normalize(mean=mean, std=std))
    # dataset_name = cfg['dataset'].pop('dataset_name')
    dataset_name = 'CULane'
    Dataset_Type = getattr(CULane_dataset_random, dataset_name)
    train_dataset = Dataset_Type(args.data_dir, "train", transform_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg['dataset']['batch_size'], shuffle=True,
                              collate_fn=train_dataset.collate, num_workers=6)
    transform_val_img = Resize(resize_shape)
    transform_val_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
    transform_val = Compose(transform_val_img, transform_val_x)

    valid_dataset = Dataset_Type(args.data_dir, 'val', transform_val)
    valid_loader = DataLoader(valid_dataset, batch_size=4, collate_fn=valid_dataset.collate, num_workers=4)
    return train_loader, valid_loader


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

            save_name = os.path.join(self.args.s_weight + '/StudentV1_{}'.format(self.args.round) + '.pth')
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
            save_name = os.path.join(self.args.s_weight + '/StudentV1_{}'.format(self.args.round) + '.pth')
            copy_name = os.path.join(self.args.s_weight + '/StudentV1_{}'.format(self.args.round) + '_best.pth')
            shutil.copyfile(save_name, copy_name)


def main():
    set_seed()
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # model_t = _TNET(backbone='regnet_y_3_2gf', pretrained=True)
    # model_t.to(device0)
    model_s = NET(backbone='regnet_y_400mf', pretrained=True)
    model_s.to(device1)

    config = '/home/arc-cjl2902/LaneDetection2/Student/config.json'
    with open(config) as f:
        cfg = json.load(f)
    print(cfg)
    transforms_val = Resize(tuple(cfg['dataset']['resize_shape']))
    """
    Start Training
    """
    # for i in range(2, 44):
    #     print('\n-------------Start {}-------------\n'.format(i))
    args = parse_args(102)
    if not os.path.exists(args.t_weight) or os.path.exists(args.s_weight) or os.path.exists(args.kd_weight):
        os.mkdir(args.t_weight)
        os.mkdir(args.s_weight)
        os.mkdir(args.kd_weight)

    train_loader, valid_loader = datasets(args, cfg)
    """
    Student
    """
    record_s = SummaryWriter(args.logdir_s)
    base_lr = 1e-3
    min_lrs = 1e-8
    total = 10 * len(train_loader)
    warmup = int(total * 0.1)
    max_iter = int(total * 0.8)
    # optimizer_s = optim.SGD(model_s.parameters(), **cfg['optim'])
    optimizer_s = optim.Adam(model_s.parameters(), base_lr)
    # optimizer_s = optim.SGD(model_s.parameters(), base_lr)
    # lr_s = lr_scheduler.PolyLR(optimizer_s, 0.8, **cfg['lr_scheduler'])
    lr_s = lr_scheduler.PolyLR(optimizer_s, 0.9, max_iter=max_iter, min_lrs=min_lrs, warmup=warmup)
    best_val_loss_s = 1e6
    scaler_s = GradScaler()
    student = STUDENT(device1, args, cfg, train_loader, valid_loader, transforms_val, record_s, model_s,
                      optimizer_s, lr_s, best_val_loss_s, scaler_s)
    for epoch in range(0, cfg['MAX_EPOCHS']):
        student.train(epoch)
        if epoch % 1 == 0:
            print("\nValidation for experiment:", args.s_weight)
            print(time.strftime('%H:%M:%S', time.localtime()))
            student.valid(epoch)


if __name__ == '__main__':
    main()
