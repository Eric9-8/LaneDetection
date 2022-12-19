import json
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from utils import my_regnet
from tqdm import tqdm
from utils.transforms import *
import time
import test_dataset
import torch
import random
import os
from test2 import encoder
import torchvision.models as models


def set_seed(seed=42):  # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
map1 = torch.rand(46, 288, 800, 2)
map2 = torch.rand(46, 288, 800, 2)
map3 = torch.rand(46, 288, 800, 2)
path = '/home/arc-cjl2902/Dataset/50_CULaneDataset'
with open('./config.json') as f:
    cfg = json.load(f)
resize_shape = tuple(cfg['dataset']['resize_shape'])
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = Compose(Resize(resize_shape), ToTensor(), Normalize(mean=mean, std=std))
dataset = test_dataset.DataTest(path, 'set0', transform)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate, num_workers=8, shuffle=False)


def euclidean_distance(p, q):
    return abs(p.cpu() - q.cpu()).sum()


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


def read_img(path):
    file_path = os.path.join(path, 'list', 're.txt')
    image = []
    # seg_label = []
    # exist = []
    img_name = []
    label_name = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            l = line.split(' ')
            img = cv2.imread(os.path.join(path, l[0][1:]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).type(torch.float) / 255.
            # label = cv2.imread(os.path.join(path, l[1][1:]))[:, :, 0]
            # torch.from_numpy(label).type(torch.long)
            # ex = np.array([int(x) for x in l[2:]])
            image.append(img)
            # seg_label.append(label)
            # exist.append(ex)
            img_name.append(os.path.join(path, l[0][1:]))
            label_name.append(os.path.join(path, l[1][1:]))

    # sample = {
    #     'image': image,
    #     # 'seg_label': np.array(seg_label),
    #     # 'exist': np.array(exist),
    #     'image_name': img_name,
    #     'label_name': label_name
    # }
    return image, img_name, label_name


def diversity(img):
    k = 4
    B = len(img)
    DIS = torch.zeros((B, B)).to(device0)
    count = torch.zeros(B).to(device0)

    s1 = time.time()
    for i in range(B):
        x_i = img[i]
        for j in range(i + 1, B):
            x_j = img[j]
            DIS[i][j] = torch.dist(x_i, x_j)

    i_lower = np.tril_indices(B, -1)
    DIS[i_lower] = DIS.T[i_lower]
    e1 = time.time()
    print('DIS {:.5f}s: '.format(e1 - s1))

    s2 = time.time()
    disk, idx = torch.sort(DIS)
    Disk = disk[:, k]
    e2 = time.time()
    print('Sort {:.5f}s: '.format(e2 - s2))

    s3 = time.time()
    for i in range(B):
        count[i] = torch.numel(DIS[i][DIS[i] < Disk])
        # for j in range(B):
        #     if DIS[i][j] < Disk[j]:
        #         count[i] += 1
        #     else:
        #         continue
    e3 = time.time()
    print('Count {:.5f}s: '.format(e3 - s3))

    # s4 = time.time()
    # save_dir = os.path.join('./', 'record_val.txt')
    #
    # with open(save_dir, 'w') as f:
    #     for i in range(B):
    #         print("{} {} {}".format(img_name[i][0], label[i][0], count[i]), end="\n", file=f)
    # e4 = time.time()
    # print('record {:.5f}s: '.format(e4 - s4))

    return count


def uncertainty(map1, map2, map3):
    global dss, dst
    SS = []
    ST = []
    for i in range(2):
        dss = []
        dst = []
        map_kd = map1[..., i + 1]
        for j in range(2):
            map_t = map2[..., j + 1]
            map_s = map3[..., j + 1]
            ss = torch.dist(map_kd, map_s).sum()
            st = torch.dist(map_kd, map_t).sum()

            dss.append(ss)
            dst.append(st)
    SS.append(min(dss))
    ST.append(min(dst))

    Ucer = (max(SS) + max(ST)) * max(max(ST) / max(SS), max(SS) / max(ST))
    return Ucer


def score(b, map_t, map_kd, map_s, path, beta):
    start0 = time.time()
    Uncer = uncertainty(map_t, map_kd, map_s)
    end0 = time.time()
    print('Calculate uncertainty {:.5f}s: '.format(end0 - start0))

    start1 = time.time()
    img, name, label = read_img(path)
    # dataset = ReadImg(img, name, label)
    end1 = time.time()
    print('load image {:.5f}s: '.format(end1 - start1))
    # print('dict size:', sys.getsizeof(dataset))

    start2 = time.time()
    Diver_map = diversity(img)
    end2 = time.time()
    print('diversity size: ', sys.getsizeof(Diver_map))
    print('Calculate diversity {:.5f}s: '.format(end2 - start2))

    Score = Uncer + beta * Diver_map[b]

    return Score


if __name__ == '__main__':
    # set_seed()
    # model = Encoder('regnet_y_400mf', pretrained=False)
    # save_name = os.path.join('/home/arc-cjl2902/LaneDetection2/Student',
    #                          'CULane_400mf_15epoch_dilation_half_train(do)' + '_best.pth')
    # save_dict = torch.load(save_name, map_location='cpu')
    # model.load_state_dict(save_dict['net'], strict=False)
    #
    # for name, paras in model.named_parameters():
    #     paras.requires_grad = False
    #
    # model.to(device0)
    # model.eval()
    encoder = encoder(device=device0)
    progress = tqdm(range(len(dataloader)))
    with torch.no_grad():
        f_map = []
        name = []
        label = []
        for batch, sample in enumerate(dataloader):
            image = sample['image'].to(device0)
            img_name = sample['image_name']
            label_name = sample['label_name']
            feature = encoder(image)
            # feature = model(image)
            f_map.append(feature)
            name.append(img_name)
            label.append(label_name)
            progress.set_description("load data {}".format(batch))
            progress.update(1)

        print('Set Already')
        # thread1 = myThread(1, 'Thread-1', f_map, name, label)
        # thread2 = myThread(2, 'Thread-2', f_map, name, label)
        # thread1.start()
        # thread2.start()

        diversity(tuple(f_map))
        print('=====================')

    progress.close()
    print('-----------------------')
    # confu = np.random.rand(5, 5)
    # confu = confu[1:, 1:]
    # print('==')
    # SC = score(0, map1, map2, map3, path, 0.3)
    # print('===========================')
