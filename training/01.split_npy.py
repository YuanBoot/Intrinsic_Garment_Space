
import numpy as np
import os

from tqdm import tqdm
from common import setting as SET

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
