import json
import matplotlib.pyplot as plt
import lr_scheduler
import torch.optim as optim
import torch
import os
import random

# _lr = 1e-3
# min_lrs = 1e-8
# warmup = int(28090 * 0.1)
# max_iter = int(28090 * 0.8)
# if __name__ == '__main__':
#     optimizer = optim.SGD([torch.ones(55)], _lr)
#     lr = lr_scheduler.PolyLR(optimizer, 0.8, max_iter=max_iter, min_lrs=min_lrs, warmup=warmup,
#                             )
#     lr_ = []
#     for cur_iter in range(28090):
#         lr__ = optimizer.param_groups[0]['lr']
#         lr_.append(lr__)
#         # lr_ += lr.get_lr()
#         optimizer.step()
#         lr.step()
#     xx = lr_[-1]
#     x = list(range(len(lr_)))
#     plt.figure(figsize=(12, 7))
#     plt.plot(x, lr_, 'r')
#     plt.xlabel('iters', size=15)
#     plt.ylabel('lr', size=15)
#     plt.show()
file = os.path.join('/home/arc-cjl2902/Dataset/50_CULaneDataset', 'list', 'train_gt_re_base.txt')
with open(file, 'r') as f:
    lines = f.readlines()
index = list(range(len(lines)))
random.shuffle(index)
rdm = os.path.join('/home/arc-cjl2902/Dataset/50_CULaneDataset', 'list', 'train_gt_do_random.txt')
with open(rdm, 'a') as k:
    for i in range(4500):
        print("{}".format(lines[index[i]].strip('\n')), file=k)
