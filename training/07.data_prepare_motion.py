
import numpy as np
import random

from common import setting as SET
from common import tools


root_joint = 0
md = [np.load(SET.JOINTS_ANIM_PATH)[:, root_joint, :]]
case_list = [1]

jointlist = SET.JOINT_LIST

uid = len(case_list)
vid = 1000

pre_frames = 20
data = np.zeros((uid*vid, pre_frames, len(jointlist)*3))
xid = 0

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
