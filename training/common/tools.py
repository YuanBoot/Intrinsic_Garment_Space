import torch
import random
import numpy as np

from common import setting as SET


def my_loss(input, target):
    return torch.sum(((input - target)**2)) / input.data.nelement()


def get_train_list():

    with open(SET.TRAIN_LIST_PATH) as f:
        content = f.readlines()

    train_list = [x.strip() for x in content]

    with open(SET.EVAL_LIST_PATH) as f:
        content = f.readlines()

    eval_list = [x.strip() for x in content]

    random.shuffle(train_list)
    random.shuffle(eval_list)

    return train_list, eval_list


def lap_operator(tmp_v, tmp_f):

    lapmat = np.zeros((len(tmp_v),len(tmp_v)))
    nn = np.zeros((len(tmp_v),len(tmp_v)))

    for f0 in tmp_f:
        ff = [f0[0]-1,f0[1]-1,f0[2]-1]
        for k0 in range(3):
            k1 = (k0+1)%3
            if nn[ff[k0],ff[k1]] == 0:
                lapmat[ff[k0],ff[k0]] = lapmat[ff[k0],ff[k0]]+1
                lapmat[ff[k1],ff[k1]] = lapmat[ff[k1],ff[k1]]+1
                lapmat[ff[k0],ff[k1]] = lapmat[ff[k0],ff[k1]]-1
                lapmat[ff[k1],ff[k0]] = lapmat[ff[k1],ff[k0]]-1
                nn[ff[k0],ff[k1]] = 1
                nn[ff[k1],ff[k0]] = 1

    vnp = np.asarray(tmp_v, dtype=np.float32)
    laptar = np.matmul(lapmat, vnp)

    tlapmat = torch.FloatTensor(lapmat).cuda()
    tlaptar = torch.FloatTensor(laptar).cuda()

    return tlapmat, tlaptar


def calculate_dist(input, target):
    return torch.sum(((input - target)**2)) / input.data.nelement()

