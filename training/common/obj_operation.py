import torch
import numpy as np
import os
import sys



def objimport(fn):
	with open(fn) as f:
		content = f.readlines()
	content = [x.strip() for x in content] 
	v = [np.array(x.split(' ')[1:]).astype(np.float).tolist() for x in content if len(x)>3 and x[0] =='v']
	f = [np.array(x.split(' ')[1:]).astype(np.int).tolist() for x in content if len(x)>3 and x[0] =='f']	
	return v,f

def objimport_v(fn):
	with open(fn) as f:
		content = f.readlines()
	content = [x.strip() for x in content] 
	v = [np.array(x.split(' ')[1:]).astype(np.float).tolist() for x in content if len(x)>3 and x[0] =='v' and x[1] != 'n']
	return v
	
	
def objexport(v,f,fn):
	file = open(fn,'w')
	for xv in v:
		file.write('v '+str(xv[0])+' '+str(xv[1])+' '+str(xv[2])+'\n')
	for xf in f:
		file.write('f '+str(xf[0])+' '+str(xf[1])+' '+str(xf[2])+'\n')
	file.close()
	
