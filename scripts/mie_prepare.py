
import numpy as np
import os
import torch
import random

from torch.autograd import Variable
from tqdm import tqdm
from basis_model import *

from common import obj_operation as obj
from common import setting as SET
from common import tools

# net
net_name = "lap_err_.relu6.w=1.0.lr=0.0001.batchseq=[40, 100, 100, 200, 400, 400, 800, 800, 1600, 1600].dim=[4524, 1800, 500, 120, 30].dp=0.02.net"

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

alll_1 = 0
alll_2 = 0

samplenum = 1000
zs = np.zeros((samplenum,tardim[-1]))

root_joint = 0
md = [np.load(SET.JOINTS_ANIM_PATH)[:, root_joint, :]]
case_list = [1]

jointlist = SET.JOINT_LIST

uid = len(case_list)
vid = 1000

pre_frames = 20
data = np.zeros((uid*vid, pre_frames, len(jointlist)*3))
xid = 0


# Calculating basis net mean and std
for idx in tqdm(range(samplenum)):

	fr = flist[idx].split(' ')
	frid = int(fr[2])
	case_id = int(fr[0][6:])
	case_index = case_list.index(case_id)

	npy_path = os.path.join(SET.CASE, 'case_' + str(case_id).zfill(2), fr[1] + '_npy', str(frid).zfill(5)+'.npy')
	x = np.load(npy_path)

	myr = np.asarray(md[case_index][frid,:9]).reshape((3,3))
	myt = np.asarray(md[case_index][frid,9:]).reshape((1,3))

	x_ = np.dot(x-np.tile(myt,[vnum,1]),np.linalg.inv(myr))

	t = np.divide((x_ - data_mean),data_std)
	y = Variable((torch.FloatTensor(t).view(1,vnum*3)).cuda())

	z = model.g(y)
	zs[idx,:] = z.cpu().data.numpy().reshape(1,tardim[-1])
	w = model.h(z)
	u_ = np.multiply(w.cpu().data.numpy().reshape(vnum,3),data_std) + data_mean
	u = np.dot(u_, myr) + np.tile(myt, [vnum,1])

	loss_1 = tools.calculate_dist(w, y)
	loss_2 = tools.calculate_dist(torch.FloatTensor(u_), torch.FloatTensor(x_))
	
	alll_1 = alll_1 + float(loss_1.cpu())
	alll_2 = alll_2 + float(loss_2.cpu())


print(alll_1/samplenum, alll_2/samplenum)

m = np.mean(zs,axis=0)
s = np.std(zs,axis=0)

np.save(SET.BASIS_MEAN_PATH, m)
np.save(SET.BASIS_STD_PATH, s)


# Calculating motion mean and std
for fid in range(uid):

	case_id = case_list[fid]
	md = np.load(SET.JOINTS_ANIM_PATH)

	for pid in range(vid):

		range_start = 20
		frid = random.randint(range_start, md.shape[0]-1) 
		myr = np.asarray(md[frid, root_joint, :9]).reshape((3,3))
		myt = np.asarray(md[frid, root_joint, 9:]).reshape((1,3))

		for p1 in range(20): 
			for p2 in range(len(jointlist)):
				x = np.asarray(md[frid-p1,jointlist[p2],9:]).reshape((1,3))
				y = np.dot(x-myt,np.linalg.inv(myr))
				data[xid,p1,p2*3:p2*3+3] = y
		xid = xid + 1

m = np.mean(data,axis=0)
s = np.std(data,axis=0)

np.save(SET.MOTION_MEAN_PATH, m)
np.save(SET.MOTION_STD_PATH, s)
