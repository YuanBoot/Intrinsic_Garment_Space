import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


bnmom = 1e-3


class mie(nn.Module):

	def __init__(self,modim,tardim,dp):
		
		super(mie, self).__init__()
		tctrl = modim[-1]
		self.modim = modim
		self.tardim = tardim
		self.dp = dp

		self.moac = nn.ModuleList([])
		for idx in range(len(modim)-1):
			if idx == len(modim)-2:
				self.moac.append(nn.Softplus(modim[idx+1]).cuda())
			else:				
				self.moac.append(nn.PReLU(modim[idx+1]).cuda())

		self.xmoac = nn.ModuleList([])
		for idx in range(len(modim)-1):
			self.xmoac.append(nn.PReLU(modim[len(modim)-idx-2]).cuda())

		self.ac = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			self.ac.append(nn.PReLU(tardim[idx+1]).cuda())

		self.xac = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			self.xac.append(nn.PReLU(tardim[len(tardim)-idx-2]).cuda())

		self.net = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			x = nn.ModuleList([])
			for j in range(tctrl):
				x.append(nn.Linear(tardim[idx], tardim[idx+1]).cuda())
			self.net.append(x)

		self.xnet = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			x = nn.ModuleList([])
			for j in range(tctrl):
				x.append(nn.Linear(tardim[len(tardim)-idx-1], tardim[len(tardim)-idx-2]).cuda())
			self.xnet.append(x)

		self.monet = nn.ModuleList([])
		for idx in range(len(modim)-1):
			x = nn.ModuleList([])
			x.append(nn.Linear(modim[idx], modim[idx+1]).cuda())
			self.monet.append(x)

		self.xmonet = nn.ModuleList([])
		for idx in range(len(modim)-1):
			x = nn.ModuleList([])
			x.append(nn.Linear(modim[len(modim)-idx-1], modim[len(modim)-idx-2]).cuda())
			self.xmonet.append(x)

			
		self.d = nn.Dropout(p=dp)
		bnmom=1e-3

		self.bn = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			self.bn.append(nn.BatchNorm1d(tardim[idx], affine=True, momentum=bnmom).cuda())

		self.xbn = nn.ModuleList([])
		for idx in range(len(tardim)-1):
			self.xbn.append(nn.BatchNorm1d(tardim[len(tardim)-idx-2], affine=True, momentum=bnmom).cuda())

		self.mobn = nn.ModuleList([])
		for idx in range(len(modim)-1):
			self.mobn.append(nn.BatchNorm1d(modim[idx], affine=True, momentum=bnmom).cuda())

		self.xmobn = nn.ModuleList([])
		for idx in range(len(modim)-1):
			self.xmobn.append(nn.BatchNorm1d(modim[len(modim)-idx-2], affine=True, momentum=bnmom).cuda())


	def m(self, x0): #x0:bsxdatadim
		
		dobn = 1>2

		for idx in range(len(self.modim)-1):
			if dobn:
				x0 = self.mobn[idx](x0)
			x0 = self.monet[idx][0](x0)
			x0 = self.moac[idx](x0)

		return x0

	def xm(self, x0): #x0:bsxdatadim
		
		dobn = 1>2

		for idx in range(len(self.modim)-1):
			x0 = self.xmonet[idx][0](x0)
			x0 = self.xmoac[idx](x0)
			if dobn and idx < len(self.modim)-2:
				x0 = self.xmobn[idx](x0)
		return x0
		
	def g(self, x0,weights): #x0:bsxdatadim
		
		tctrl = len(weights)
		
		dobn = 1>2

		for idx in range(len(self.tardim)-1):
			if dobn:
				x0 = self.bn[idx](x0)
			tx = 0
			tx_test = 0
			w_sum = 0
			for j in range(tctrl):
				v = self.net[idx][j](x0)
				w = weights[j]/torch.sum(weights)
				w_sum += w
				wv = w*v
				tx_test = tx_test + v
				tx = tx + wv
				#tx = tx + weights[j]/torch.sum(weights) * self.net[idx][j](x0)
			x0 = tx
			x0 = self.ac[idx](x0)

		return x0

	def h(self, x0, weights): #x0:bsxdatadim

		temp = int(x0.shape[0])
		x0 = x0.reshape((1, temp))

		tctrl = len(weights)

		dobn = 1>2

		for idx in range(len(self.tardim)-1):
			tx = 0
			for j in range(tctrl):
				tx = tx + weights[j]/torch.sum(weights)* self.xnet[idx][j](x0)
			x0 = tx
			x0 = self.xac[idx](x0)
			if dobn and idx < len(self.tardim)-2:
				x0 = self.xbn[idx](x0)
			
		return x0



