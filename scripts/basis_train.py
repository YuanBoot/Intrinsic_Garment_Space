
import numpy as np
import os
import torch
import random
import glob
import sys

from datetime import datetime
from torch.autograd import Variable
import torch.optim as optim
from basis_model import *

from common import obj_operation as obj
from common import setting as SET
from common import tools


def calculate_loss(w, y, std, mean, batchsize, tlapmat, tlaptar, vnum, weight_lap):

    loss_1 = tools.calculate_dist(w*std, y*std) 
    w_ = w*std + mean
    y_ = y*std + mean

	# TODO: loss_2 function

    loss_2 = 0
    for bid in range(batchsize):
        tlaptar = torch.mm(tlapmat,y_[bid].view(vnum,3))
        loss_2 = loss_2 + tools.calculate_dist(torch.mm(tlapmat,w_[bid].view(vnum,3)),tlaptar)
		
    loss_2 = loss_2/float(batchsize) * weight_lap
    return loss_1 + loss_2, loss_1, loss_2


def get_train_data(std, mean, batchsize, flist_train, vnum, case_list):

    temp_data = np.zeros((batchsize,vnum*3))
    for bid in range(batchsize):

        fid = flist_train[random.randint(0,len(flist_train)-1)].split(' ')
        frid = int(fid[2])

        case_id = int(fid[0][6:])
        case_index = case_list.index(case_id)
        npy_path = os.path.join(SET.CASE, 'case_' + str(case_id).zfill(2), fid[1] + '_npy', str(frid).zfill(5) + '.npy')

        x = np.load(npy_path)
        myr = np.asarray(md[case_index][int(fid[2]),:9]).reshape((3,3))
        myt = np.asarray(md[case_index][int(fid[2]),9:]).reshape((1,3))
        y = np.divide((np.dot(x-np.tile(myt,[vnum,1]),np.linalg.inv(myr)) - mean), std)
        temp_data[bid,:] = y.reshape(1,vnum*3)

    return temp_data


vnum = SET.VERTS_NUM
lrate = SET.LEARNING_RATE
batchsize_eval = SET.BATCH_EVAL
batchsize_seq = SET.BATCH_SEQ
tardim = SET.TRAIN_DIM
dp = SET.DP
weight_lap = SET.WEIGHT_LAP

fname_str = 'lap_err_.relu6.w=%s.lr=%s.dim=%s.dp=%s' % (str(weight_lap), str(lrate), str(tardim), str(dp))
fname = os.path.join(SET.BASIS, fname_str)
print(fname)

model = basis(tardim,dp).cuda()
#model_name = "lap_err_.relu6.w=1.0.lr=0.0001.batchseq=[40, 100, 100, 200, 400, 400, 800, 800, 1600, 1600].dim=[4134, 1800, 500, 120, 30].dp=0.02.net"
#model.load_state_dict(torch.load(os.path.join(SET.BASIS, model_name)))
model.train()

optimizer = optim.RMSprop(model.parameters(), lr=lrate, alpha=0.99, eps=1e-8, weight_decay=0., momentum=0.1, centered=False)

data_mean = np.load(SET.DATA_MEAN_PATH)
data_std = np.load(SET.DATA_STD_PATH)

flist_train, flist_eval = tools.get_train_list()

with open(fname+'.txt', "a") as myfile:
	myfile.write('start at: '+str(datetime.now())+'\n')

root_joint = 0
md = [np.load(SET.JOINTS_ANIM_PATH)[:, root_joint, :]]
case_list = [1]

tmp_v = SET.MESH_VERTS
tmp_f = SET.MESH_FACES

tlapmat, tlaptar = tools.lap_operator(tmp_v, tmp_f)


for myiter in range(100000):
	
	optimizer.zero_grad()
	batchsize = batchsize_seq[np.amin((int(myiter / 8000),len(batchsize_seq)-1))]
	we = torch.FloatTensor(np.tile(data_std.reshape(1,vnum*3),[batchsize,1])).cuda()
	me = torch.FloatTensor(np.tile(data_mean.reshape(1,vnum*3),[batchsize,1])).cuda()

	train_data = get_train_data(data_std, data_mean, batchsize, flist_train, vnum, case_list)
	
	y = Variable((torch.FloatTensor(train_data).view(batchsize,vnum*3)).cuda())
	z = model.g(y)
	w = model.h(z)
	
	loss, loss_1 , loss_2 = calculate_loss(w, y, we, me, batchsize, tlapmat, tlaptar, vnum, weight_lap)
	loss.backward()
	optimizer.step()
	print('batch: %d loss_intrain: %.6f %.6f %.6f' % (myiter,loss.cpu(),loss_1.cpu(),loss_2.cpu()))


	if (myiter+1) % 100 == 0:
		model.eval()
		train_data = get_train_data(data_std, data_mean, batchsize, flist_train, vnum, case_list)

		y = Variable((torch.FloatTensor(train_data).view(batchsize,vnum*3)).cuda())
		z = model.g(y)
		w = model.h(z)

		loss_train, loss_train_1, loss_train_2 = calculate_loss(w, y, we, me, batchsize, tlapmat, tlaptar, vnum, weight_lap)

		batchsize = batchsize_eval
		we = torch.FloatTensor(np.tile(data_std.reshape(1,vnum*3),[batchsize,1])).cuda()
		me = torch.FloatTensor(np.tile(data_mean.reshape(1,vnum*3),[batchsize,1])).cuda()

		eval_data = get_train_data(data_std, data_mean, batchsize, flist_train, vnum, case_list)

		y = Variable((torch.FloatTensor(eval_data).view(batchsize,vnum*3)).cuda())
		z = model.g(y)
		w = model.h(z)

		loss_eval, loss_eval_1, loss_eval_2 = calculate_loss(w, y, we, me, batchsize, tlapmat, tlaptar, vnum, weight_lap)

		print('%s batch: %d loss_train: %.6f loss_eval: %.6f' % (str(datetime.now()),myiter,loss_train.cpu(),loss_eval.cpu()))
		with open(fname+'.txt', "a") as myfile:
			myfile.write(str(myiter)+' '+str(datetime.now())+' '+str(float(loss_train.cpu()))+' '+str(float(loss_eval.cpu()))+' '+str(float(loss_train_1.cpu()))+' '+str(float(loss_train_2.cpu()))+' '+str(float(loss_eval_1.cpu()))+' '+str(float(loss_eval_2.cpu()))+'\n')
		
		if (myiter+1) % 1000 == 0:
			torch.save(model.state_dict(), fname+'.net')
		model.train()

