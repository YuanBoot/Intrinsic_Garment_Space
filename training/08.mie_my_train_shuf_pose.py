
import numpy as np
import os
import sys
import torch
import random
import glob

import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable

from basis_my_arch_relu6 import *
from mie_my_arch_softplus import *

from common import setting as SET
from common import tools


vnum = SET.VERTS_NUM
lrate = SET.MIE_LERNING_RATE
batchsize_eval = SET.MIE_BATCHSIZE_EVAL
batchsize_seq_1 = SET.MIE_BATCHSIZE_01
batchsize_seq_2 = SET.MIE_BATCHSIZE_02
tardim = SET.MIE_DIM
dp = SET.DP
ws = SET.WS

joints_number = SET.JOINT_NUM
frnum = SET.PRE_FRAME
modim = SET.MOTION_DIM


fnid = 'bs_relu6'
fname_str = 'pose_%s_train_ae_shuf.%s.weight=%s.err_.lr=%s.modim=%s.tardim=%s.batchseq=%s%s.dp=%s' % (str(frnum),fnid,str(ws),str(lrate),str(modim),str(tardim),str(batchsize_seq_1),str(batchsize_seq_2),str(dp))
fname = os.path.join(SET.MIE, fname_str)

basis_tardim = SET.TRAIN_DIM
basis_name = "lap_err_.relu6.w=1.0.lr=0.0001.batchseq=[40, 100, 100, 200, 400, 400, 800, 800, 1600, 1600].dim=[4134, 1800, 500, 120, 30].dp=0.02.net"
basis_dp = SET.DP
basis_net = basis(basis_tardim,basis_dp).cuda()
basis_net.load_state_dict(torch.load(os.path.join(SET.BASIS, basis_name)))
basis_net.eval()

model = mie(modim,tardim,dp).cuda()
#mie_name = 'pose_20_train_ae_shuf.bs_relu6.weight=[1000.0, 1.0].err_.lr=0.001.modim=[1320, 480, 120, 30].tardim=[30, 120, 120].batchseq=[4, 8, 8, 8, 16, 16, 16, 16][4, 4, 4, 8, 10, 10, 10, 10].dp=0.02.net'
#model.load_state_dict(torch.load(os.path.join(SET.MIE, basis_name)))
model.train()

optimizer = optim.RMSprop(model.parameters(), lr=lrate, alpha=0.99, eps=1e-8, weight_decay=0., momentum=0.1, centered=False)

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

with open(fname+'.txt', "a") as myfile:
	myfile.write('start at: '+str(datetime.now())+'\n')

jointlist = SET.JOINT_LIST


def get_same_material_list(flist_train, batchsize):

	blst = np.random.permutation(len(flist_train))
	xlst = []
	t = 0

	for i in range(len(blst)):
		if flist_train[blst[i]][1] == flist_train[blst[0]][1]:
			xlst.append(blst[i])
			t=t+1
			if t == batchsize: break

	return xlst


def calculate_disciptor(fr):

	frid = int(fr[2])
	case_id = int(fr[0][6:])
	case_index = case_list.index(case_id)

	myr = np.asarray(md[case_index][frid, root_joint, :9]).reshape((3, 3))
	myt = np.asarray(md[case_index][frid, root_joint, 9:]).reshape((1, 3))
	cm = np.zeros((frnum, joints_number*3))

	for p1 in range(frnum):
		for p2 in range(len(jointlist)):
			temp_index = max(frid - p1, 0)
			x = np.asarray(md[case_index][temp_index, jointlist[p2], 9:]).reshape((1,3))
			y = np.dot(x-myt,np.linalg.inv(myr))
			cm[p1,p2*3:p2*3+3] = y

	cm = np.divide((cm - motion_mean),motion_std)
	cm = Variable((torch.FloatTensor(cm).view(1,frnum*joints_number*3)).cuda())
	motion_vector = model.m(cm)[0]

	npy_path = os.path.join(SET.CASE, 'case_' + str(case_id).zfill(2), fr[1] + '_npy', str(frid).zfill(5)+'.npy')
	x = np.load(npy_path)

	x_ = np.dot(x-np.tile(myt,[vnum,1]),np.linalg.inv(myr))
	t = np.divide((x_ - data_mean),data_std)
	y = Variable((torch.FloatTensor(t).view(1,vnum*3)).cuda())

	temp_net = basis_net.g(y)
	shape_vector = torch.zeros(basis_std.shape, dtype=torch.float32).cuda()
	for i in range(len(basis_std[0])):
		if basis_std[0][i] > 1e-3:
			shape_vector[0][i] = (temp_net[0][i] - basis_mean[0][i]) / basis_std[0][i]

	return shape_vector, motion_vector


def loss_update(temp_loss_list, temp_data, temp_weights, temp_zs):

	loss_1 = temp_loss_list[0]
	loss_2 = temp_loss_list[1]
	loss_3 = temp_loss_list[2]

	temp_xdata = temp_data[torch.randperm(batchsize_2)]
	loss_1 = loss_1 + tools.calculate_dist(temp_data,temp_xdata)

	for bid in range(batchsize_2):
		q = model.h(temp_data[bid],temp_weights[bid])
		loss_2 = loss_2 + tools.calculate_dist(q,temp_zs[bid])
		q = model.h(temp_xdata[bid],temp_weights[bid])
		loss_3 = loss_3 + tools.calculate_dist(q,temp_zs[bid])

	return [loss_1, loss_2, loss_3]


def deal_with_loss(temp_loss_list, temp_ws, bs_1, bs_2):

	loss_1 = temp_ws[0]*temp_loss_list[0] / float(bs_1)
	loss_2 = temp_ws[1]*temp_loss_list[1] / float(bs_1) / float(bs_2)
	loss_3 = temp_ws[1]*temp_loss_list[2] / float(bs_1) / float(bs_2)
	loss = loss_1 + loss_2

	return loss, loss_1, loss_2, loss_3


def train(bs_1, bs_2, temp_flist):

	loss_list = [0.0 ,0.0 ,0.0]

	for bidx in range(bs_1):

		xlst = get_same_material_list(temp_flist, bs_2)

		data = torch.zeros([bs_2,tardim[-1]]).cuda()
		zs = torch.zeros([bs_2,tardim[0]]).cuda()
		weights = torch.zeros([bs_2,modim[-1]]).cuda()

		for bid in range(bs_2):

			fr = temp_flist[xlst[bid]].split(' ')
			z, w = calculate_disciptor(fr)

			data[bid] = model.g(z, w)
			zs[bid] = z
			weights[bid] = w

		loss_list = loss_update(loss_list, data, weights, zs)

	return deal_with_loss(loss_list, ws, bs_1, bs_2)


for myiter in range(50000):

	optimizer.zero_grad()

	batchsize_1 = batchsize_seq_1[np.amin((int(myiter / 1000),len(batchsize_seq_1)-1))]
	batchsize_2 = batchsize_seq_2[np.amin((int(myiter / 1000),len(batchsize_seq_2)-1))]

	loss, loss_1, loss_2, loss_3 = train(batchsize_1, batchsize_2, flist_train)

	loss.backward()
	optimizer.step()
	print('batch: %d loss_intrain: %.6f %.6f %.6f' % (myiter,loss.cpu(),loss_1.cpu(),loss_2.cpu()))

	if (myiter+1) % 100 == 0:
		model.eval()
		
		batchsize_1 = 6
		batchsize_2 = batchsize_eval

		loss_train, loss_1t, loss_2t, loss_3t = train(batchsize_1, batchsize_2, flist_train)
		loss_eval, loss_1e, loss_2e, loss_3e  = train(batchsize_1, batchsize_2, flist_eval)

		st = str(myiter)+' '+str(datetime.now())+' '+str(float(loss_train.cpu()))+' '+str(float(loss_eval.cpu()))+' '+str(float(loss_1t.cpu())/ws[0])+' '+str(float(loss_2t.cpu())/ws[1])+' '+str(float(loss_3t.cpu())/ws[1])+' '+str(float(loss_1e.cpu())/ws[0])+' '+str(float(loss_2e.cpu())/ws[1])+' '+str(float(loss_3e.cpu())/ws[1])+' '+str(ws)+'\n'
		print(st)
		with open(fname+'.txt', "a") as myfile:
			myfile.write(st)
			
		if (myiter+1) % 500 == 0:
			torch.save(model.state_dict(), fname+'.net')

		model.train()

