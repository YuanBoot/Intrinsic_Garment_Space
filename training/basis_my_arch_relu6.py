import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class basis(nn.Module):

	def __init__(self,tardim,dp):
		super(basis, self).__init__()
		self.tardim = tardim
		self.dp = dp

		self.ac = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			if idx == len(tardim)-2:
				self.ac.append(nn.ReLU6().cuda())	
			else:
				self.ac.append(nn.PReLU(tardim[idx+1]).cuda())

		self.xac = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			self.xac.append(nn.PReLU(tardim[len(tardim)-idx-2]).cuda())

		self.net = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			self.net.append(nn.Linear(tardim[idx], tardim[idx+1]).cuda())

		self.xnet = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			self.xnet.append(nn.Linear(tardim[len(tardim)-idx-1], tardim[len(tardim)-idx-2]).cuda())
		
		self.d = nn.Dropout(p=dp)
		bnmom=1e-3

		self.bn = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			self.bn.append(nn.BatchNorm1d(tardim[idx], affine=True, momentum=bnmom).cuda())

		self.xbn = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			self.xbn.append(nn.BatchNorm1d(tardim[len(tardim)-idx-2], affine=True, momentum=bnmom).cuda())

		
	def g(self, x0): #x0:bsxdatadim
		
		dobn = 1>2

		for idx in range(len(self.tardim)-1):
			if dobn:
				x0 = self.bn[idx](x0)
			x0 = self.net[idx](x0)
			x0 = self.ac[idx](x0)

		return x0

	def h(self, x0): #x0:bsxdatadim
		
		dobn = 1>2

		for idx in range(len(self.tardim)-1):
			x0 = self.xnet[idx](x0)
			x0 = self.xac[idx](x0)
			if dobn and idx < len(self.tardim)-2:
				x0 = self.xbn[idx](x0)
			
		return x0


























