
import numpy as np
import os

from common import obj_operation as obj
from common import setting as SET
from tqdm import tqdm

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

		
				

		
