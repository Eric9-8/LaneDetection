import os
import random

import numpy as np


def subset(alist, idxs):
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])
    return sub_list


def random_split(path, image_set):
    file_list = os.path.join(path, 'list', '{}_gt_re.txt'.format(image_set))
    with open(file_list, 'r') as f:
        lines = f.readlines()
    index = list(range(len(lines)))
    random.shuffle(index)
    elem_num = len(lines) // 44
    for idx in range(44):
        start, end = idx * elem_num, (idx + 1) * elem_num
        sub_list = subset(lines, index[start:end])
        with open('{}_gt_re.txt'.format('set' + str(idx)), 'w') as f:
            for _ in range(len(sub_list)):
                f.write(sub_list[_])

    print('=====')
    # with open('{}_gt_do.txt'.format(image_set), 'w') as do, open('{}_gt_re.txt'.format(image_set), 'w') as re:
    #     for _ in range(len(lines) // 2):
    #         do.write(lines.pop(random.randint(0, len(lines) - 1)))
    #     re.writelines(lines)


def merge_dataset(base_file, set_record):
    pass


if __name__ == '__main__':
    # random_split('/home/arc-cjl2902/Dataset/50_CULaneDataset', 'train')
    pass
