
import numpy as np
import os
import random

from tqdm import tqdm
from common import setting as SET

fn1 = os.path.join(SET.INFO_BASIS, SET.TRAIN_LIST)
fn2 = os.path.join(SET.INFO_BASIS, SET.EVAL_LIST)

f1 = open(fn1, 'w')
f2 = open(fn2, 'w')

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
