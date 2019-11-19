import numpy as np
import os
import torch
import random
import sys
import glob

import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable
from basis_my_arch_relu6 import *

from common import obj_operation as obj
from common import setting as SET
from common import tools


# net
net_name = "lap_err_.relu6.w=1.0.lr=0.0001.batchseq=[40, 100, 100, 200, 400, 400, 800, 800, 1600, 1600].dim=[4134, 1800, 500, 120, 30].dp=0.02.net"

# Training Parameters
vnum = SET.VERTS_NUM
tardim = SET.TRAIN_DIM
dp = SET.DP
fname = os.path.join(SET.BASIS, net_name)

data_mean = np.load(SET.DATA_MEAN_PATH)
data_std = np.load(SET.DATA_STD_PATH)

flist_train, flist_eval = tools.get_train_list()
flist = flist_train + flist_eval
random.shuffle(flist)

model = basis(tardim, dp).cuda()
model.load_state_dict(torch.load(fname))
model.eval()

tmp_v = SET.MESH_VERTS
tmp_f = SET.MESH_FACES

root_joint = 0
md = [np.load(SET.JOINTS_ANIM_PATH)[:, root_joint, :]]
case_list = [1]

alll_1 = 0
alll_2 = 0

samplenum = 10

for idx in range(samplenum):

	fid = flist[idx].split(' ')
	frid = int(fid[2])
	case_id = int(fid[0][6:])
	case_index = case_list.index(case_id)

	npy_path = os.path.join(SET.CASE, 'case_' + str(case_id).zfill(2), fid[1] + '_npy', str(frid).zfill(5)+'.npy')
	x = np.load(npy_path)

	myr = np.asarray(md[case_index][int(fid[2]),:9]).reshape((3,3))
	myt = np.asarray(md[case_index][int(fid[2]),9:]).reshape((1,3))
	x_ = np.dot(x-np.tile(myt,[vnum,1]),np.linalg.inv(myr))
	t = np.divide((x_ - data_mean),data_std)

	y = Variable((torch.FloatTensor(t).view(1,vnum*3)).cuda())
	z = model.g(y)
	w = model.h(z)
	u_ = np.multiply(w.cpu().data.numpy().reshape(vnum,3),data_std)+data_mean
	u = np.dot(u_,myr)+np.tile(myt,[vnum,1])

	loss_1 = tools.my_loss(w,y)
	loss_2 = tools.my_loss(torch.FloatTensor(u_),torch.FloatTensor(x_))
	
	alll_1 = alll_1 + float(loss_1.cpu())
	alll_2 = alll_2 + float(loss_2.cpu())

	file_name_grt = 'eval_%s_%s_%s_%s_grt.obj' % (str(idx).zfill(3), fid[0], fid[1], fid[2])
	file_name_rec = 'eval_%s_%s_%s_%s_%s_rec.obj' % (str(idx).zfill(3), fid[0], fid[1], fid[2], str(float(loss_2.cpu())))

	obj.objexport(x, tmp_f, os.path.join(SET.EVAL_BASIS, file_name_grt))
	obj.objexport(u, tmp_f, os.path.join(SET.EVAL_BASIS, file_name_rec))


print(alll_1/samplenum, alll_2/samplenum)
