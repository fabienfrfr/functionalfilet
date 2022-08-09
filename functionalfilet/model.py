#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:43:17 2022
@author: fabien
"""
# ML modules
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F

# system module
import pickle, datetime, os

# networks construction
from graph.GRAPH_EAT import GRAPH_EAT
from net.pRNN_GEN import pRNN

##### Prerequisite
from collections import deque
import itertools

class ReplayMemory(object):
	def __init__(self, capacity, named_tuple):
		self.Transition = named_tuple
		self.memory = deque([],maxlen=capacity)
	def push(self, *args):
		"""Save a transition"""
		self.memory.append(self.Transition(*args))
	def sample(self, batch_size):
		last = self.__len__()
		sample = list(itertools.islice(self.memory, last-batch_size, last))
		return sample
		#return random.sample(self.memory, batch_size)
	def __len__(self):
		return len(self.memory)

class CTRL_NET(nn.Module):
	def __init__(self, IO):
		super(CTRL_NET, self).__init__()
		I,O = IO
		if I+O > 64 :
			H = 2*int(np.sqrt(I+O))
		else : 
			H = 16
		self.IN = nn.Conv1d(I, I, 1, groups=I, bias=True)
		self.H1 = nn.Linear(I, H)
		self.H2 = nn.Linear(H, H)
		self.OUT = nn.Linear(H, O)

	def forward(self, x):
		s = x.shape
		x = self.IN(x.view(s[0],s[1],1)).view(s)
		x = F.relu(x)
		x = F.relu(self.H1(x))
		x = F.relu(self.H2(x))
		return self.OUT(x)

##### FF MODULE
"""  
Note hybrid propriety :   
If GEN = 0, equivalent of no evolution during training : only SGD
if NB_BATCH > NB_BATCH/GEN, equivalent of no SGD : only evolution
"""
class FunctionnalFillet():
	def __init__(self, arg, NAMED_MEMORY=None, TYPE="class", DEVICE=True, TIME_DEPENDANT = False):
		# parameter
		self.IO =  arg[0]
		self.BATCH = arg[1]
		self.NB_GEN = arg[2]
		self.NB_SEEDER = arg[3]
		self.NB_EPISOD = arg[4]
		self.ALPHA = arg[5] # 1-% of predict (not random step)
		self.NB_E_P_G = int(self.NB_EPISOD/self.NB_GEN)
		self.TIME_DEP = TIME_DEPENDANT
		self.TYPE = TYPE
		self.NAMED_M = NAMED_MEMORY
		if DEVICE==True :
			self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		else :
			self.DEVICE = DEVICE
		# generate first ENN model
		self.GRAPH_LIST = [GRAPH_EAT([self.IO, 1], None) for n in range(self.NB_SEEDER-1)]
		self.SEEDER_LIST = [CTRL_NET(self.IO)]
		for g in self.GRAPH_LIST :
			NEURON_LIST = g.NEURON_LIST
			self.SEEDER_LIST += [pRNN(NEURON_LIST, self.BATCH, self.IO[0], self.DEVICE, STACK=self.TIME_DEP)]
		# training parameter
		self.NEURON_LIST = []
		self.update_model()
		# selection
		self.loss = pd.DataFrame(columns=['GEN','IDX_SEED', 'EPISOD', 'N_BATCH', 'LOSS_VALUES'])
		self.supp_param = None
		# evolution param
		self.NB_CONTROL = int(np.power(self.NB_SEEDER, 1./4))
		self.NB_EVOLUTION = int(np.sqrt(self.NB_SEEDER)-1) # square completion
		self.NB_CHALLENGE = int(self.NB_SEEDER - (self.NB_EVOLUTION*(self.NB_EVOLUTION+1) + self.NB_CONTROL))
		# evolution variable
		self.PARENTING = [-1*np.ones(self.NB_SEEDER)[None]]
		self.PARENTING[0][0][:self.NB_CONTROL] = 0
		
	def update_model(self):
		# neuron graph history
		self.NEURON_LIST += [g.NEURON_LIST for g in self.GRAPH_LIST]
		# torch
		self.optimizer = [torch.optim.Adam(s.parameters()) for s in self.SEEDER_LIST]
		if self.TYPE == "class" :
			self.criterion = [nn.CrossEntropyLoss() for n in range(self.NB_SEEDER)]
		else :
			self.criterion = [nn.SmoothL1Loss() for n in range(self.NB_SEEDER)] # regression / RL
		# memory
		if self.NAMED_M == None :
			self.memory = {"X_train":None, "Y_train":None, "X_test":None, "Y_test":None}
		else :
			self.memory = [ReplayMemory(1024, self.NAMED_M) for n in range(self.NB_SEEDER)]
		
	def step(self, INPUT, index=0) :
		in_tensor = torch.tensor(INPUT, dtype=torch.float)
		out_probs = self.SEEDER_LIST[index](in_tensor)
		# exploration dilemna
		DILEMNA = np.squeeze(out_probs.detach().numpy())
		if DILEMNA.sum() == 0 or str(DILEMNA.sum()) == 'nan' :
			out_choice = np.random.randint(self.IO[1])
		else :
			if DILEMNA.min() < 0 : DILEMNA = DILEMNA-DILEMNA.min() # order garanty
			## ADD dispersion between near values (ex : q-table, values is near)
			order = np.argsort(DILEMNA)+1
			#order[np.argmax(order)] += 1
			order = np.exp(order)
			# probability
			p_norm = order/order.sum()
			out_choice = np.random.choice(self.IO[1], p=p_norm)
		return out_choice
	
	def predict(self, INPUT, index=0):
		if isinstance(INPUT, torch.Tensor) :
			in_tensor = INPUT.type(torch.float)
		else :
			in_tensor = torch.tensor(INPUT, dtype=torch.float)
		# device
		in_tensor = in_tensor.to(self.DEVICE)
		# extract prob
		out_probs = self.SEEDER_LIST[index](in_tensor).cpu().detach().numpy()
		return np.argmax(out_probs, axis=1)
	
	def train(self, output, target, generation=0, index=0, episod=0, i_batch=0):
		# reset
		#self.optimizer[index].zero_grad()
		self.SEEDER_LIST[index].zero_grad()
		# loss computation
		loss = self.criterion[index](output, target)
		# do back-ward
		loss.backward()
		self.optimizer[index].step()
		# save loss
		self.loss = self.loss.append({'GEN':generation,
									  'IDX_SEED':index,
									  'EPISOD':episod,
									  'N_BATCH':i_batch,
									  'LOSS_VALUES':float(loss.detach().numpy())},
									  ignore_index=True)
	
	def selection(self, GEN, supp_factor=1):
		# sup median loss selection
		TailLoss = np.ones(self.NB_SEEDER)
		# extract data
		sub_loss = self.loss[self.loss.GEN == GEN]
		# verify if you have SDG (only evolution selection)
		if sub_loss.size > 0 :
			gb_seed = sub_loss.groupby('IDX_SEED')
			# sup median loss selection
			for i,g in gb_seed :
				if self.ALPHA != 1 :
					Tail_eps = g.EPISOD.min()+(g.EPISOD.max() - g.EPISOD.min())*self.ALPHA
				else :
					Tail_eps = g.EPISOD.median()
				TailLoss[int(i)] = g[g.EPISOD > Tail_eps].LOSS_VALUES.mean()
			# normalization
			relativeLOSS = (TailLoss-TailLoss.min())/(TailLoss.max()-TailLoss.min())
		else :
			relativeLOSS = TailLoss
		# coeffect, belong to [0,3]
		score = supp_factor + supp_factor*relativeLOSS + relativeLOSS
		# order
		order = np.argsort(score[self.NB_CONTROL:])
		### stock control network
		NET_C = self.SEEDER_LIST[:self.NB_CONTROL]
		### generation parenting
		PARENT = [0]*self.NB_CONTROL
		### survivor
		GRAPH_S = []
		NET_S = []
		GRAPH_IDX = list(order[:self.NB_EVOLUTION])
		for i in GRAPH_IDX :
			GRAPH_S += [self.GRAPH_LIST[i]]
			if np.random.choice((True,False), 1, p=[1./self.NB_GEN,1-1./self.NB_GEN]):
				NET_S += [self.SEEDER_LIST[self.NB_CONTROL:][i]]
			else :
				NET_S += [pRNN(GRAPH_S[-1].NEURON_LIST, self.BATCH, self.IO[0], STACK=self.TIME_DEP)]
			PARENT += [i+1]
		### mutation
		GRAPH_M = []
		NET_M = []
		for g,j in zip(GRAPH_S,GRAPH_IDX):
			for i in range(self.NB_EVOLUTION):
				GRAPH_M += [g.NEXT_GEN()]
				NET_M += [pRNN(GRAPH_M[-1].NEURON_LIST, self.BATCH, self.IO[0], STACK=self.TIME_DEP)]
				PARENT += [j+1]
		### news random
		GRAPH_N = []
		NET_N = []
		for n in range(self.NB_CHALLENGE):
			GRAPH_N += [GRAPH_EAT([self.IO, 1], None)]
			NET_N += [pRNN(GRAPH_N[-1].NEURON_LIST, self.BATCH, self.IO[0], STACK=self.TIME_DEP)]
			PARENT += [-1]
		### update seeder list and stock info
		self.PARENTING += [np.array(PARENT)[None]]
		self.GRAPH_LIST = GRAPH_S + GRAPH_M + GRAPH_N
		self.SEEDER_LIST = NET_C + NET_S + NET_M + NET_N
		### update model
		self.UPDATE_MODEL()

	def fit(self, train_in, train_target, test_in, test_target):
		# SUPERVISED TRAIN
		return

	def finalization(self, supp_param=None, save=True):
		self.PARENTING = np.concatenate(self.PARENTING).T
		self.supp_param = supp_param
		if save :
			if(not os.path.isdir('OUT')): os.makedirs('OUT')
			time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			filehandler = open("OUT"+os.path.sep+"MODEL_FF_"+time+".obj", 'wb')
			pickle.dump(self, filehandler); filehandler.close()
	
	def load(self, Filename):
		with open('OUT'+os.path.sep+Filename, 'rb') as f:
			return pickle.load(f)

### basic exemple
if __name__ == '__main__' :
	import torchvision
	mnist_data = torchvision.datasets.MNIST('', download=True)
	## parameter
	IO =  (784,10)
	BATCH = 25
	NB_GEN = 100
	NB_SEED = 5**2
	NB_EPISODE = 25000 #25000
	ALPHA = 0.9

	## load model
	model = FunctionnalFillet([IO, BATCH, NB_GEN, NB_SEED, NB_EPISODE, ALPHA], TYPE='RL')

	## predict (note : linearize image !)
	N = BATCH
	X, Y = mnist_data.train_data, mnist_data.train_labels
	x, y = X[:N].reshape(N,-1), Y[:N]
	y_pred = model.predict(x,1)

