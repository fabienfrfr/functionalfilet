# 20220817 fabienfrfr

import numpy as np
import torch, torch.nn as nn

# networks construction
from graph_eat import GRAPH_EAT
from pRNN_net import pRNN

# utils
from functionalfilet.utils import CTRL_NET
from graph_show import Graphic

class EvoNeuralNet(nn.Module, Graphic):
	def __init__(self, IO=(64,16), BATCH=25, DEVICE=torch.device('cpu'), control=False, invert=False):
		super().__init__()
		# parameter
		self.io = IO
		self.batch = BATCH
		self.device = DEVICE
		self.invert = invert
		# Input adjustment
		self.patch_in = lambda x:x
		# Block
		if control == False :
			# graph
			self.graph = GRAPH_EAT(self.io)
			self.net = self.graph.NEURON_LIST
			# pRNN block
			self.enn_block = pRNN(self.net, self.batch, self.io[0], self.device)
		else :
			# control net (add possibility to add own model)
			self.enn_block = CTRL_NET(self.io, self.device)
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
		self.I, self.O = I,O
		if I < O & self.invert == False :
			print("[INFO] Input is lower than output and INVERT is false, the adaptation of the evolutionary block I/O can be aberrant..")
		# Input part
		if I != self.io[0] :
			self.patch_in = self.patch(I, self.io[0])
		# Output part
		if O != self.io[1]:
			self.patch_out = self.patch(self.io[1], O, first=False)
			self.fc = nn.Linear(O,O)

	def forward(self,x):
		s = x.shape
		# input
		x = self.patch_in(x.view(s[0],s[1],1)).view(s[0],self.io[0])
		# enn block
		x = self.enn_block(x)
		# output
		x = self.patch_out(x.view(s[0],self.io[1],1)).view(s[0],self.O)
		x = self.fc(x)
		return x

### basic exemple
if __name__ == '__main__' :
	model = EvoNeuralNet()
	model.checkIO(128, 10)

	x = torch.rand(5,128)
	y = torch.randint(0, 8, (5,))

	y_pred = model(x)