import os
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.models as models


# 1.首先要加载n张图像，并生成diversity的矩阵
# 2.然后与uncertainty计算得分


def set_seed(seed=42):  # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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


def encoder(device):
    set_seed()
    model = Encoder('regnet_y_400mf', pretrained=False)
    save_name = os.path.join('/home/arc-cjl2902/LaneDetection2/Student/exp/v1',
                             'StudentV1' + '_best.pth')
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


# map1 = [x for x in torch.rand((49, 288, 800, 2))]
# map2 = [x for x in torch.rand((49, 288, 800, 2))]
# map3 = [x for x in torch.rand((49, 288, 800, 2))]
# map4 = [x for x in torch.rand((49, 288, 800, 2))]
# map5 = [x for x in torch.rand(49)]
# map6 = [x for x in torch.rand(49)]


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


# un = uncertain(map1, map2, map3)
# di = div(map4)
# name, label = score(un, di, map5, map6)
# print("========================")
