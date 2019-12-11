
import numpy as np
import os
import random
from tqdm import tqdm
from common import setting as SET
from common import obj_operation as obj


def split_npy():

	data_folder = os.path.join(SET.CASE, SET.CASE_01)
	folder_list = os.listdir(data_folder)

	for fnx in folder_list:

		flx = os.listdir(os.path.join(data_folder, fnx))

		for fny in tqdm(flx):

			npy_file = os.path.join(data_folder, fnx, fny)
			p1 = np.load(npy_file)
			t = fny.split('_')
			q1 = int(t[2])
			q2 = int(t[3].split('.')[0])
			ct = -1

			for idx in range(q1,q2+1):
				ct = ct + 1
				save_file_name = str(idx).zfill(5) + '.npy'
				np.save(os.path.join(data_folder, fnx, save_file_name), p1[ct,:,:])

			os.remove(npy_file)


def make_data_list():

	f1 = open(os.path.join(SET.INFO_BASIS, SET.TRAIN_LIST), 'w')
	f2 = open(os.path.join(SET.INFO_BASIS, SET.EVAL_LIST), 'w')

	case_list = os.listdir(SET.CASE)

	for case in case_list:
		fl = os.listdir(os.path.join(SET.CASE, case))
		for fnx in fl:
			fn = fnx[:-4]
			flist = os.listdir(os.path.join(SET.CASE, case, fnx))
			for uid in range(len(flist)):
				t = np.random.rand()
				if t < .9:
					f1.write(str(case)+' '+fn+' '+str(uid)+'\n')
				else:
					f2.write(str(case)+' '+fn+' '+str(uid)+'\n')

	f1.close()
	f2.close()


def visualize_skeleton():

	md = np.load(SET.JOINTS_ANIM_PATH)

	xid = 10
	ts = [-1,1]

	fs = [[1,2,3],[3,2,4],[7,6,5],[6,7,8],[1,3,5],[5,3,7],[2,6,4],[4,6,8],[3,4,7],[7,4,8],[1,5,2],[2,5,6]]

	for tid in range(md.shape[1]):
		myt = np.asarray(md[xid,tid,9:]).reshape((1,3))
		vs = [[0,0,0]] *8
		ct = -1
		for p1 in ts:
			for p2 in ts:
				for p3 in ts:
					ct = ct + 1
					vs[ct] = [myt[0,0]+p1, myt[0,1]+p2, myt[0,2]+p3]

		file_name = "fr_%s.sk_%s.obj" % (str(xid).zfill(5), str(tid).zfill(5))
		obj.objexport(vs, fs, os.path.join(SET.SKELETON, file_name))


def calculate_mean_and_std():

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


def main():
	split_npy()
	make_data_list()
	#visualize_skeleton()
	calculate_mean_and_std()


if __name__ == '__main__':
    main()