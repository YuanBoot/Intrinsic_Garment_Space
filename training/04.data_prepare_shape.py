
import numpy as np
import os
import random

from tqdm import tqdm
from common import obj_operation as obj
from common import setting as SET


vnum = SET.VERTS_NUM
case_folder_list = os.listdir(SET.CASE)


md = [np.load(os.path.join(SET.JOINTS_ANIM_PATH))]
case_list = [1]

with open(os.path.join(SET.INFO_BASIS, SET.TRAIN_LIST)) as f:
	content = f.readlines()

flist_train = [x.strip().split(' ') for x in content]

random.shuffle(flist_train)

uid = 4000

verts = np.zeros((uid,vnum,3))

for fid in tqdm(range(uid)):
	fr = flist_train[fid]
	frid = int(fr[2])
	case_id = int(fr[0][6:])
	case_index = case_list.index(case_id)
	root_joint = 0
	myr = np.asarray(md[case_index][frid,root_joint,:9]).reshape((3,3))
	myt = np.asarray(md[case_index][frid,root_joint,9:]).reshape((1,3))
	
	npy_path = os.path.join(SET.CASE, case_folder_list[case_id-1], fr[1] + '_npy', str(frid).zfill(5)+'.npy')
	x = np.load(npy_path)

	y = np.dot(x-np.tile(myt,[vnum,1]),np.linalg.inv(myr))
	verts[fid,:,:] = y.reshape(1,vnum,3)


for i in range(40):
	xid = random.randint(0,uid)
	obj.objexport(verts[xid,:,:], SET.MESH_FACES, os.path.join(SET.DATA_SAMPLE, "s_%s.obj" % str(i)))


m = np.mean(verts,axis=0)
s = np.std(verts,axis=0)

np.save(os.path.join(SET.INFO_BASIS, SET.DATA_MEAN), m)
np.save(os.path.join(SET.INFO_BASIS, SET.DATA_STD), s)


