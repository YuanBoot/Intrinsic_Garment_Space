
import numpy as np
import os
import torch
import random
import sys
from datetime import datetime

from torch.autograd import Variable
import torch.optim as optim

from basis_model import *
from mie_model import *

from common import obj_operation as obj
from common import setting as SET
from common import tools


vnum = SET.VERTS_NUM
tardim = SET.MIE_DIM
dp = SET.DP
ws = SET.WS

jointlist = SET.JOINT_LIST
joints_number = SET.JOINT_NUM
frnum = SET.PRE_FRAME
modim = SET.MOTION_DIM

basis_tardim = SET.TRAIN_DIM
basis_name = "lap_err_.relu6.w=1.0.lr=0.0001.batchseq=[40, 100, 100, 200, 400, 400, 800, 800, 1600, 1600].dim=[4134, 1800, 500, 120, 30].dp=0.02.net"
basis_dp = SET.DP
basis_net = basis(basis_tardim,basis_dp).cuda()
basis_net.load_state_dict(torch.load(os.path.join(SET.BASIS, basis_name)))
basis_net.eval()

model = mie(modim,tardim,dp).cuda()
mie_name = 'pose_20_train_ae_shuf.bs_relu6.weight=[1000.0, 1.0].err_.lr=0.001.modim=[1320, 480, 120, 30].tardim=[30, 120, 120].batchseq=[4, 8, 8, 8, 16, 16, 16, 16][4, 4, 4, 8, 10, 10, 10, 10].dp=0.02.net'
model.load_state_dict(torch.load(os.path.join(SET.MIE, mie_name)))
model.eval()

data_mean = np.load(SET.DATA_MEAN_PATH).reshape(vnum,3)
data_std = np.load(SET.DATA_STD_PATH).reshape(vnum,3)
motion_mean = np.load(SET.MOTION_MEAN_PATH).reshape(20,joints_number*3)[0,:]
motion_std = np.load(SET.MOTION_STD_PATH).reshape(20,joints_number*3)[0,:]
basis_mean = torch.FloatTensor(np.load(SET.BASIS_MEAN_PATH).reshape(1,basis_tardim[-1])).cuda()
basis_std = torch.FloatTensor(np.load(SET.BASIS_STD_PATH).reshape(1,basis_tardim[-1])).cuda()
motion_std[motion_std<1e-5] = 1.

root_joint = 0
md = [np.load(SET.JOINTS_ANIM_PATH)]
case_list = [1]

flist_train, flist_eval = tools.get_train_list()
flist_all = flist_train + flist_eval

tmp_v = SET.MESH_VERTS
tmp_f = SET.MESH_FACES

batchsize_1 = 50
batchsize_2 = 1
loss_1 = 0
loss_2 = 0


mylist = np.random.randint(len(flist_eval),size=100)


for bidx in range(batchsize_1):

	xlst = [mylist[bidx]]
	data = torch.zeros([batchsize_2,tardim[-1]]).cuda()
	zs = torch.zeros([batchsize_2,tardim[0]]).cuda()
	weights = torch.zeros([batchsize_2,modim[-1]]).cuda()

	for bid in range(batchsize_2):

		fr = flist_eval[xlst[bid]].split(' ')
		frid = int(fr[2])
		case_id = int(fr[0][6:])
		case_index = case_list.index(case_id)

		myr = np.asarray(md[case_index][frid, root_joint, :9]).reshape((3, 3))
		myt = np.asarray(md[case_index][frid, root_joint, 9:]).reshape((1, 3))
		cm = np.zeros((frnum, joints_number * 3))

		for p1 in range(frnum):
			for p2 in range(len(jointlist)):
				temp_index = max(frid - p1, 0)
				x = np.asarray(md[case_index][temp_index, jointlist[p2], 9:]).reshape((1, 3))
				y = np.dot(x - myt, np.linalg.inv(myr))
				cm[p1, p2 * 3:p2 * 3 + 3] = y

		cm = np.divide((cm - motion_mean), motion_std)
		cm = Variable((torch.FloatTensor(cm).view(1, frnum * joints_number * 3)).cuda())
		w = model.m(cm)[0]

		npy_path = os.path.join(SET.CASE, 'case_' + str(case_id).zfill(2), fr[1] + '_npy', str(frid).zfill(5)+'.npy')
		x = np.load(npy_path)

		x_ = np.dot(x - np.tile(myt, [vnum, 1]), np.linalg.inv(myr))

		t = np.divide((x_ - data_mean),data_std)
		y = Variable((torch.FloatTensor(t).view(1,vnum*3)).cuda())

		z = torch.div(basis_net.g(y)-basis_mean,basis_std)

		res = z.clone()
		res[z != z] = 0.0
		z = res

		temp_data = model.g(z,w)
		data[bid] = temp_data
		zs[bid] = z
		weights[bid] = w

	loss_1 = loss_1 + torch.sum(data.std(dim=0))
	dmean = data.mean(dim=0)

	for bid in range(batchsize_2):

		q = model.h(dmean,weights[bid])
		w = basis_net.h(torch.mul(q,basis_std)+basis_mean)
		u_ = np.multiply(w.cpu().data.numpy().reshape(vnum,3),data_std)+data_mean
		u = np.dot(u_,(myr))+np.tile(myt,[vnum,1])
		loss_2 = tools.calculate_dist(q,zs[bid])

		file_name_grt = 'eval_%s_%s_%s_%s_grt.obj' % (str(bid).zfill(3), fr[0], fr[1], fr[2])
		file_name_rec = 'eval_%s_%s_%s_%s_%s_rec.obj' % (str(bid).zfill(3), fr[0], fr[1], fr[2], str(float(loss_2.cpu())))

		obj.objexport(x, tmp_f, os.path.join(SET.EVAL_MIE, file_name_grt))
		obj.objexport(u, tmp_f, os.path.join(SET.EVAL_MIE, file_name_rec))








