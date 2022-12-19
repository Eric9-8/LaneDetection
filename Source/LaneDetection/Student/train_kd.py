"""
    单独训练蒸馏模型
"""
import os
import argparse
import json
from tqdm import tqdm
import shutil
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from utils import CULane_dataset, lr_scheduler
from utils.transforms import *
from model_KD import _KDNET
from Teacher.model import _TNET

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default='./config.json')
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--logdir", type=str, default='./new_log/KDv1_1')
    parser.add_argument("--weight_dir", type=str, default='/home/arc-cjl2902/LaneDetection2/Student/exp/KDv1_2')
    parser.add_argument("--data_dir", type=str, default='/home/arc-cjl2902/Dataset/50_CULaneDataset')
    parser.add_argument("--teacher_weight", type=str, default='/home/arc-cjl2902/LaneDetection2/Teacher/exp/v1_2')
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
device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

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
valid_loader = DataLoader(valid_dataset, batch_size=4, collate_fn=valid_dataset.collate, num_workers=8)

# load student_kd
model_kd = _KDNET(backbone='regnet_y_400mf')

# load teacher
model_T = _TNET(backbone='regnet_y_3_2gf', pretrained=False)
save_name = os.path.join(args.teacher_weight, 'TeacherV1_{}'.format(args.round) + '_best.pth')
save_dict = torch.load(save_name, map_location='cpu')
print('\nloading', save_name, '------------- From Epoch:', save_dict['epoch'])
model_T.load_state_dict(save_dict['net'])

for name, paras in model_T.named_parameters():
    paras.requires_grad = False

model_kd.to(device0)
model_T.to(device0)
model_T.eval()

# Training set
optimizer = optim.SGD(model_kd.parameters(), **cfg['optim'])
lr_scheduler = lr_scheduler.PolyLR(optimizer, 0.8, **cfg['lr_scheduler'])
best_val_loss = 1e6
scaler = GradScaler()


def train(epoch):
    print("Train epoch is {}".format(epoch))
    model_kd.train()
    train_loss = 0
    # train_loss_seg_gt = 0
    # train_loss_exist_gt = 0
    # train_loss_seg_kd = 0
    # train_loss_exist_kd = 0

    progress = tqdm(range(len(train_loader)))

    for batch, sample in enumerate(train_loader):
        img = sample['image'].to(device0)
        seg_label = sample['seg_label'].to(device0)
        seg_label = torch.where(seg_label == 2, 1, seg_label)
        seg_label = torch.where(seg_label == 3, 1, seg_label)
        seg_label = torch.where(seg_label == 4, 1, seg_label)
        # exist = sample['exist'].to(device0)

        # seg_T, exist_T = model_T(img.to(device0))[:2]
        seg_T = model_T(img.to(device0))[0]
        seg_kd = seg_T.to(device=device0)
        # seg_kd_label.to(device=device0)
        # exist_kd = exist_T.to(device=device0)
        seg_kd_label = F.softmax(seg_kd, dim=1)

        optimizer.zero_grad()
        with autocast():
            seg_pre, loss = model_kd(img, seg_label, seg_kd_label)
            # seg_pre, exist_pre, loss_seg_gt, loss_exist_gt, loss_seg_kd, loss_exist_kd, loss = model_kd(img,
            #                                                                                             seg_label,
            #                                                                                             exist,
            #                                                                                             seg_kd_label,
            #                                                                                             exist_kd)
        scaler.scale(loss).backward()
        # loss.backward()
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()
        lr_scheduler.step()

        iter_idx = epoch * len(train_loader) + batch
        train_loss = loss.item()
        # train_loss_seg_gt = loss_seg_gt.item()
        # train_loss_exist_gt = loss_exist_gt.item()
        # train_loss_seg_kd = loss_seg_kd.item()
        # train_loss_exist_kd = loss_exist_kd.item()
        progress.set_description("batch loss {:.3f}:".format(loss.item()))
        progress.update(1)

        lr = optimizer.param_groups[0]['lr']
        record.add_scalar('/train/train_loss', train_loss, iter_idx)
        # record.add_scalar('/train/train_loss_seg_gt', train_loss_seg_gt, iter_idx)
        # record.add_scalar('/train/train_loss_exist_gt', train_loss_exist_gt, iter_idx)
        # record.add_scalar('/train/train_loss_seg_kd', train_loss_seg_kd, iter_idx)
        # record.add_scalar('/train/train_loss_exist_kd', train_loss_exist_kd, iter_idx)
        record.add_scalar('learning_rate', lr, iter_idx)

    progress.close()
    record.flush()

    if epoch % 1 == 0:
        save_dict = {
            "epoch": epoch,
            "net": model_kd.module.state_dict() if isinstance(model_kd,
                                                              torch.nn.DataParallel) else model_kd.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss
        }
        save_name = os.path.join(args.weight_dir + '/StudentKDV1_{}'.format(args.round) + '.pth')
        torch.save(save_dict, save_name)
        print("model is saved: {}".format(save_name))

    print("==============================\n")


def valid(epoch):
    global best_val_loss
    print("Valid epoch is {}".format(epoch))

    model_kd.eval()
    val_loss = 0
    # val_loss_seg_gt = 0
    # val_loss_exist_gt = 0
    # val_loss_seg_kd = 0
    # val_loss_exist_kd = 0
    progress = tqdm(range(len(valid_loader)))

    with torch.no_grad():
        for batch, sample in enumerate(valid_loader):
            img = sample['image'].to(device0)
            seg_label = sample['seg_label'].to(device0)
            seg_label = torch.where(seg_label == 2, 1, seg_label)
            seg_label = torch.where(seg_label == 3, 1, seg_label)
            seg_label = torch.where(seg_label == 4, 1, seg_label)
            # exist = sample['exist'].to(device0)

            # seg_T, exist_T = model_T(img.to(device0))[:2]
            seg_T = model_T(img.to(device0))[0]
            seg_kd = seg_T.to(device=device0)
            # seg_kd_label.to(device=device0)
            # exist_kd = exist_T.to(device=device0)
            seg_kd_label = F.softmax(seg_kd, dim=1)

            seg_pre, loss = model_kd(img, seg_label, seg_kd_label)

            # seg_pre, exist_pre, loss_seg_gt, loss_exist_gt, loss_seg_kd, loss_exist_kd, loss = model_kd(img,
            #                                                                                             seg_label,
            #                                                                                             exist,
            #                                                                                             seg_kd_label,
            #                                                                                             exist_kd)

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

                    color = np.array([255, 125, 0])
                    color_mask = np.argmax(seg_pre[b], axis=0)
                    lane_img[color_mask == 1] = color[0]

                    # color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]])

                    # color_mask = np.argmax(seg_pre[b], axis=0)
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
            # val_loss_seg_gt += loss_seg_gt.item()
            # val_loss_exist_gt += loss_exist_gt.item()
            # val_loss_seg_kd += loss_seg_kd.item()
            # val_loss_exist_kd += loss_exist_kd.item()

            progress.set_description("batch loss {:.3f}".format(loss.item()))
            progress.update(1)

    progress.close()

    iter_idx = (epoch + 1) * len(train_loader) + batch
    record.add_scalar('valid/valid_loss', val_loss, iter_idx)
    # record.add_scalar('valid/valid_loss_seg_gt', val_loss_seg_gt, iter_idx)
    # record.add_scalar('valid/valid_loss_exist_gt', val_loss_exist_gt, iter_idx)
    # record.add_scalar('valid/valid_loss_seg_kd', val_loss_seg_kd, iter_idx)
    # record.add_scalar('valid/valid_loss_exist_kd', val_loss_exist_kd, iter_idx)
    record.flush()

    print("==============================\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_name = os.path.join(args.weight_dir + '/StudentKDV1_{}'.format(args.round) + '.pth')
        copy_name = os.path.join(args.weight_dir + '/StudentKDV1_{}'.format(args.round) + '_best.pth')
        shutil.copyfile(save_name, copy_name)


def main():
    global best_val_loss
    if args.resume:
        save_dict = torch.load(
            os.path.join(args.weight_dir + '/StudentKDV1_{}'.format(args.round) + '.pth'))
        if isinstance(model_kd, torch.nn.DataParallel):
            model_kd.module.load_state_dict(save_dict['net'])
        else:
            model_kd.load_state_dict(save_dict['net'])
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
