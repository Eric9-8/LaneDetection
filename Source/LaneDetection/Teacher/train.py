import os
import argparse
import json
from tqdm import tqdm
import shutil
import time

import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from torchstat import stat
import torch.distributed as dist
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from Student.utils import CULane_dataset, lr_scheduler
from Student.utils.transforms import *
from model import _TNET

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default='./config.json')
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--logdir", type=str, default='./new_log/TeacherV1_2')
    parser.add_argument("--weight_dir", type=str, default='/home/arc-cjl2902/LaneDetection2/Teacher/exp/v1_2')
    parser.add_argument("--data_dir", type=str, default='/home/arc-cjl2902/Dataset/50_CULaneDataset')
    parser.add_argument("--round", type=int, default=2)
    args = parser.parse_args()
    return args


args = parse_args()
record = SummaryWriter(args.logdir)
config = args.config_dir
with open(config) as f:
    cfg = json.load(f)
print(cfg)
if not os.path.exists(args.weight_dir):
    os.mkdir(args.weight_dir)
# pars
resize_shape = tuple(cfg['dataset']['resize_shape'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)

# Train dataset
transform_train = Compose(Resize(resize_shape), Rotation(2), ToTensor(), Normalize(mean=mean, std=std))
dataset_name = cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(CULane_dataset, dataset_name)
train_dataset = Dataset_Type(args.data_dir, "train", transform_train)

# Half train dataset
# random_seed = 42
# data_size = len(train_dataset)
# indices = list(range(data_size))
# np.random.seed(random_seed)
# np.random.shuffle(indices)
# train_indices = indices[:int(np.floor(0.5 * data_size))]
# remain_indices = indices[int(np.floor(0.5 * data_size)):]
# train_sample = SubsetRandomSampler(train_indices)
# train_loader = DataLoader(train_dataset, batch_size=cfg['dataset']['batch_size'],
#                           collate_fn=train_dataset.collate, sampler=train_sample, num_workers=8)
# Full train dataset
train_loader = DataLoader(train_dataset, batch_size=cfg['dataset']['batch_size'], shuffle=True,
                          collate_fn=train_dataset.collate, num_workers=4)
# Valid dataset
transform_val_img = Resize(resize_shape)
transform_val_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform_val = Compose(transform_val_img, transform_val_x)

valid_dataset = Dataset_Type(args.data_dir, 'val', transform_val)

# valid_sampler = distributed.DistributedSampler(valid_dataset)
valid_loader = DataLoader(valid_dataset, batch_size=4, collate_fn=valid_dataset.collate, num_workers=4)

model = _TNET(backbone='regnet_y_3_2gf', pretrained=True)
# stat(model, (3, 288, 800))
model.to(device)
# model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

optimizer = optim.SGD(model.parameters(), **cfg['optim'])
lr_scheduler = lr_scheduler.PolyLR(optimizer, 0.8, **cfg['lr_scheduler'])
best_val_loss = 1e6

# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

scaler = GradScaler()


def train(epoch):
    print("Train epoch is {}".format(epoch))
    model.train()
    train_loss = 0
    # train_loss_seg = 0
    # train_loss_exist = 0
    progress = tqdm(range(len(train_loader)))

    for batch, sample in enumerate(train_loader):
        img = sample['image'].to(device)
        seg_label = sample['seg_label'].to(device)
        # exist = sample['exist'].to(device)
        seg_label = torch.where(seg_label == 2, 1, seg_label)
        seg_label = torch.where(seg_label == 3, 1, seg_label)
        seg_label = torch.where(seg_label == 4, 1, seg_label)

        optimizer.zero_grad()
        with autocast():
            # seg_pre, exist_pre, loss_seg, loss_exist, loss = model(img, seg_label, exist)
            seg_pre, loss = model(img, seg_label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        lr_scheduler.step()
        scaler.update()

        iter_idx = epoch * len(train_loader) + batch
        train_loss = loss.item()
        # train_loss_seg = loss_seg.item()
        # train_loss_exist = loss_exist.item()
        progress.set_description("batch loss {:.3f}:".format(loss.item()))
        progress.update(1)

        lr = optimizer.param_groups[0]['lr']
        record.add_scalar('/train/train_loss', train_loss, iter_idx)
        # record.add_scalar('/train/train_loss_seg', train_loss_seg, iter_idx)
        # record.add_scalar('/train/train_loss_exist', train_loss_exist, iter_idx)
        record.add_scalar('learning_rate', lr, iter_idx)

    progress.close()
    record.flush()

    if epoch % 1 == 0:
        save_dict = {
            "epoch": epoch,
            "net": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss
        }
        save_name = os.path.join(args.weight_dir + '/TeacherV1_{}'.format(args.round) + '.pth')
        torch.save(save_dict, save_name)
        print("model is saved: {}".format(save_name))

    print("==============================\n")


def valid(epoch):
    global best_val_loss
    print("Valid epoch is {}".format(epoch))

    model.eval()
    val_loss = 0
    # val_loss_seg = 0
    # val_loss_exist = 0
    progress = tqdm(range(len(valid_loader)))

    with torch.no_grad():
        for batch, sample in enumerate(valid_loader):
            img = sample['image'].to(device)
            seg_label = sample['seg_label'].to(device)
            seg_label = torch.where(seg_label == 2, 1, seg_label)
            seg_label = torch.where(seg_label == 3, 1, seg_label)
            seg_label = torch.where(seg_label == 4, 1, seg_label)
            # exist = sample['exist'].to(device)

            # seg_pre, exist_pre, loss_seg, loss_exist, loss = model(img, seg_label, exist)
            seg_pre, loss = model(img, seg_label)
            gap_num = 5
            if batch % gap_num == 0 and batch < 50 * gap_num:
                origin_img = []
                seg_pre = seg_pre.detach().cpu().numpy()
                # exist_pre = exist_pre.detach().cpu().numpy()

                for b in range(len(img)):
                    img_name = sample['image_name'][b]
                    img = cv2.imread(img_name)
                    img = transform_val_img({'image': img})['image']

                    lane_img = np.zeros_like(img)
                    # color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]])
                    color = np.array([255, 125, 0])

                    color_mask = np.argmax(seg_pre[b], axis=0)
                    # for i in color_mask:
                    #     if i > 0:
                    #         lane_img[color_mask] = color[0]
                    lane_img[color_mask == 1] = color[0]
                    # for i in range(4):
                    #     if exist_pre[b, i] > 0.5:
                    #         lane_img[color_mask == (i + 1)] = color[i]

                    img = cv2.addWeighted(src1=lane_img, alpha=0.7, src2=img, beta=1., gamma=0.)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
                    # cv2.putText(lane_img, "{}".format([1 if exist_pre[b, i] > 0.5 else 0 for i in range(4)]), (20, 20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                    origin_img.append(img)
                    origin_img.append(lane_img)
                record.add_images("img{}".format(batch), torch.tensor(np.array(origin_img)), epoch, dataformats='NHWC')

            val_loss += loss.item()
            # val_loss_seg += loss_seg.item()
            # val_loss_exist += loss_exist.item()

            progress.set_description("batch loss {:.3f}".format(loss.item()))
            progress.update(1)

    progress.close()

    iter_idx = (epoch + 1) * len(train_loader) + batch
    record.add_scalar('valid/valid_loss', val_loss, iter_idx)
    # record.add_scalar('valid/valid_loss_seg', val_loss_seg, iter_idx)
    # record.add_scalar('valid/valid_loss_exist', val_loss_exist, iter_idx)
    record.flush()

    print("==============================\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_name = os.path.join(args.weight_dir + '/TeacherV1_{}'.format(args.round) + '.pth')
        copy_name = os.path.join(args.weight_dir + '/TeacherV1_{}'.format(args.round) + '_best.pth')
        shutil.copyfile(save_name, copy_name)


def main():
    global best_val_loss
    if args.resume:
        save_dict = torch.load(os.path.join(args.weight_dir + '/TeacherV1_{}'.format(args.round) + '.pth'))
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(save_dict['net'])
        else:
            model.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        start_epoch = save_dict['epoch'] + 1
        best_val_loss = save_dict['best_val_loss', 1e6]
    else:
        start_epoch = 0

    # cfg['MAX_EPOCHS'] = int(np.ceil(cfg['lr_scheduler']['max_iter'] / len(train_loader)))
    print('start_epoch: ', start_epoch)
    print('MAX_EPOCHS: ', cfg['MAX_EPOCHS'])
    for epoch in range(start_epoch, cfg['MAX_EPOCHS']):
        train(epoch)
        if epoch % 1 == 0:
            print("\nValidation for experiment:", args.weight_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            valid(epoch)


if __name__ == '__main__':
    main()
