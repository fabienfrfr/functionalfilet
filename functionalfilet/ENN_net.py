# 20220817 fabienfrfr

import numpy as np
import torch, torch.nn as nn

import copy
from collections import namedtuple

# networks construction
from functionalfilet.graph_eat import GRAPH_EAT
from functionalfilet.pRNN_net import pRNN

# utils
from functionalfilet.utils import CTRL_NET

class EvoNeuralNet(nn.Module):
	def __init__(self, IO=(64,16), BATCH=25, DEVICE=torch.device('cpu'), control=False, invert=False, stack=False, graph=None, net=None):
		super().__init__()
		# parameter
		self.io = IO
		self.batch = BATCH
		self.device = DEVICE
		self.invert = invert
		self.stack = stack
		self.control = control
		# Input adjustment
		self.patch_in = lambda x:x
		# Block
		if control == False :
			# graph
			if graph == None :
				self.graph = GRAPH_EAT(self.io)
			elif net == None :
				self.graph = graph
			else :
				self.graph = GRAPH_EAT(None, NET=net)
			self.net = self.graph.NEURON_LIST
			# pRNN block
			self.enn_block = pRNN(self.net, self.batch, self.io[0], self.device, self.stack)
		else :
			# control net (add possibility to add own model)
			self.enn_block = CTRL_NET(self.io, self.device)
			self.graph = namedtuple('graph',('LIST_C'))
			self.graph.LIST_C = "None"
			self.net = self.enn_block.net
		# Output adjustment
		self.patch_out = lambda x:x
		# final layers
		self.fc = lambda x:x

	def patch(self, I,O, first=True) :
		r = int(np.rint(max(I,O)/min(I,O)))
		if first :
			block = nn.Sequential(	*[nn.Conv1d(I,O, kernel_size=r,stride=r, padding=r),
									 nn.Dropout(0.9), nn.BatchNorm1d(num_features=O),
									 nn.ReLU(), nn.MaxPool1d(kernel_size=r)]).to(self.device)
		else :
			block = nn.Sequential(	*[nn.ReLU(), nn.Conv1d(I,O, kernel_size=r,stride=r, padding=r), nn.ReLU(),
									 nn.AvgPool1d(kernel_size=r)]).to(self.device)
		return block

	def checkIO(self, I, O):
		self.RealIO = I,O
		if I < O & self.invert == False :
			print("[INFO] Input is lower than output and INVERT is false, the adaptation of the evolutionary block I/O can be aberrant..")
		# Input part
		if I != self.io[0] :
			self.patch_in = self.patch(I, self.io[0])
		# Output part
		if O != self.io[1]:
			self.patch_out = self.patch(self.io[1], O, first=False)
			self.fc = nn.Linear(O,O).to(self.device)

	def forward(self,x):
		s = x.shape
		# input
		x = self.patch_in(x.view(s[0],s[1],1)).view(s[0],self.io[0])
		# enn block
		x = self.enn_block(x)
		# output
		x = self.patch_out(x.view(s[0],self.io[1],1)).view(s[0],self.RealIO[1])
		x = self.fc(x)
		return x

	def update(self, mode):
		# copy model
		state = self.state_dict().copy()
		ann = EvoNeuralNet(self.io, self.batch, self.device, invert=self.invert, stack=self.stack, graph=copy.deepcopy(self.graph))
		ann.checkIO(*self.RealIO)
		ann.load_state_dict(state)
		# condition mode
		if mode == 'copy' :
			return ann
		elif mode == 'reset':
			ann.enn_block = pRNN(self.net, self.batch, self.io[0], self.device, self.stack)
			return ann
		elif mode == 'mut' :
			ann.graph = ann.graph.NEXT_GEN()
			ann.net = ann.graph.NEURON_LIST
			ann.enn_block = pRNN(ann.net, self.batch, self.io[0], self.device, self.stack)
			return ann
		else :
			print("[INFO] Error")

### basic exemple
if __name__ == '__main__' :
	model = EvoNeuralNet()
	model.checkIO(128, 10)

	x = torch.rand(5,128)
	y = torch.randint(0, 8, (5,))

	y_pred = model(x)

	model.graph.SHOW_GRAPH(LINK_LAYERS=False)