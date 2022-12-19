import numpy as np
import cv2
import torch
import os


# class SCORE:
#     def __init__(self, map_t, map_kd, map_s, path):
#         self.map_t = map_t
#         self.map_kd = map_kd
#         self.map_s = map_s
#         self.path = path
#
#     def euclidean(self, p, q):
#         return np.sqrt(np.square(p - q).sum())

def euclidean_distance(p, q):
    return np.sqrt(np.square(p - q).sum())


def uncertainty(seg_t, seg_kd, seg_s):
    global dss, dst
    SS = []
    ST = []
    for i in range(4):
        dss = []
        dst = []
        map_kd = seg_kd[..., i + 1]
        for j in range(4):
            map_t = seg_t[..., j + 1]
            map_s = seg_s[..., j + 1]
            dss.append(euclidean_distance(map_kd, map_s))
            dst.append(euclidean_distance(map_kd, map_t))
    SS.append(min(dss))
    ST.append(min(dst))
    Ucer = (max(SS) + max(ST)) * max(max(ST) / max(SS), max(SS) / max(ST))
    return Ucer


def read_img(path):
    file_path = os.path.join(path, 'list', 'train_gt_re.txt')
    image = []
    seg_label = []
    exist = []
    img_name = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            l = line.split(' ')
            img = cv2.imread(os.path.join(path, l[0][1:]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).type(torch.float) / 255.
            label = cv2.imread(os.path.join(path, l[1][1:]))[:, :, 0]
            torch.from_numpy(label).type(torch.long)
            ex = np.array([int(x) for x in l[2:]])
            image.append(img)
            seg_label.append(label)
            exist.append(ex)
            img_name.append(os.path.join(path, l[0][1:]))
    sample = {
        'image': image,
        'seg_label': np.array(seg_label),
        'exist': np.array(exist),
        'image_name': img_name
    }
    return sample


def diversity(dataset):
    k = 5
    image = torch.stack(dataset['image'])
    B, _, _, _ = image.size()
    DIS = np.zeros((B, B))
    Disk = np.zeros((B, 1))
    Div = np.zeros((B, 1), dtype=int)
    # image = image.view(B, -1)
    for i in range(B):
        x_i = image[i]
        for j in range(B):
            x_j = image[j]
            d = euclidean_distance(x_i, x_j)
            DIS[i][j] = d
    for i in range(B):
        Disk[i] = sorted(DIS[i])[k]
    for i in range(B):
        for j in range(B):
            if DIS[i][j] < Disk[j]:
                Div[i] += 1
            else:
                continue
    return Div


def score(b, map_t, map_kd, map_s, path, beta):
    Uncer = uncertainty(map_t, map_kd, map_s)
    dataset = read_img(path)
    Diver_map = diversity(dataset)
    Score = Uncer + beta * Diver_map[b]
    return Score
