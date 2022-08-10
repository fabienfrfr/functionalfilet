#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-08-09
@author: fabienfrfr
"""
# ML modules
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# system module
import pickle, datetime
import os, time
from tqdm import tqdm

# networks construction
from graph.GRAPH_EAT import GRAPH_EAT
from net.pRNN_GEN import pRNN

# utils
from utils import ReplayMemory, CTRL_NET

##### FF MODULE
"""  
Note hybrid propriety :   
If GEN = 0, equivalent of no evolution during training : only SGD
if NB_BATCH > NB_BATCH/GEN, equivalent of no SGD : only evolution
"""
class FunctionalFilet():
	def __init__(self, arg, NAMED_MEMORY=None, TYPE="class", DEVICE=True, TIME_DEPENDANT = False):
		print("[INFO] Starting System...")
		# parameter
		self.IO =  arg[0]
		self.BATCH = arg[1]
		self.NB_GEN = arg[2]
		self.NB_SEEDER = int(np.rint(np.sqrt(arg[3]))**2)
		self.ALPHA = arg[4] # 1-% of predict (not random step)
		if TYPE == "class" :
			self.NB_BATCH = int(arg[5] / self.BATCH)  # nb_batch = (dataset_lenght * nb_epoch) / batch_size
			self.NB_E_P_G = int(self.NB_BATCH/self.NB_GEN)
		else :
			self.NB_EPISOD = arg[5]
			self.NB_E_P_G = int(self.NB_EPISOD/self.NB_GEN)
		self.TIME_DEP = TIME_DEPENDANT
		self.TYPE = TYPE
		self.NAMED_M = NAMED_MEMORY
		if DEVICE==True :
			self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		else :
			self.DEVICE = DEVICE
		print("[INFO] Calculation type : " + self.DEVICE.type)
		# generate first ENN model
		print("[INFO] Generate first evolutionnal neural networks..")
		print("[INFO] Graph part..")
		self.GRAPH_LIST = [GRAPH_EAT([self.IO, 1], None) for n in range(self.NB_SEEDER-1)]
		self.SEEDER_LIST = [CTRL_NET(self.IO, self.DEVICE)]
		print("[INFO] Networks part..")
		for g in self.GRAPH_LIST :
			NEURON_LIST = g.NEURON_LIST
			self.SEEDER_LIST += [pRNN(NEURON_LIST, self.BATCH, self.IO[0], self.DEVICE, STACK=self.TIME_DEP)]
		print("[INFO] ENN Generated!")
		# training parameter
		print("[INFO] Generate training parameters for population..")
		self.NEURON_LIST = []
		self.update_model()
		print("[INFO] Generate selection parameters for population..")
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
		print("[INFO] Model created!")
		
	def update_model(self):
		# neuron graph history
		self.NEURON_LIST += [g.NEURON_LIST for g in self.GRAPH_LIST]
		# torch
		self.optimizer = [torch.optim.Adam(s.parameters()) for s in self.SEEDER_LIST]
		if self.TYPE == "class" :
			self.criterion = [nn.CrossEntropyLoss().to(self.DEVICE) for n in range(self.NB_SEEDER)]
		else :
			self.criterion = [nn.SmoothL1Loss().to(self.DEVICE) for n in range(self.NB_SEEDER)] # regression / RL
		# memory
		if self.NAMED_M == None :
			self.memory = {"X_train":None, "Y_train":None, "X_test":None, "Y_test":None}
		else :
			self.memory = [ReplayMemory(1024, self.NAMED_M) for n in range(self.NB_SEEDER)]
		
	def step(self, INPUT, index=0, message=False):
		in_tensor = torch.tensor(INPUT, dtype=torch.float)
		if message : print("[INFO] Switch to inference mode for model:"+str(index))
		"""
		Note : For Reinforcement Q learning, it's better to have 2 paired model, this to avoid switching between "train" and "eval" mode during training & optimize convergence. Like :
		- target_net.load_state_dict(policy_net.state_dict())
		- target_net.eval()
		It wasn't done here, because it's not a big problem during training without dropout or batch norm (eval ~= train)
		"""
		self.SEEDER_LIST[index].eval()
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
		if message : print("[INFO] Chosen prediction : " + str(out_choice))
		return out_choice
	
	def predict(self, INPUT, index=0, message=False):
		if isinstance(INPUT, torch.Tensor) :
			in_tensor = INPUT.type(torch.float)
		else :
			in_tensor = torch.tensor(INPUT, dtype=torch.float)
		# device
		if message : print("[INFO] Switch to inference mode for model:"+str(index))
		in_tensor = in_tensor.to(self.DEVICE)
		# extract prob
		self.SEEDER_LIST[index].eval()
		out_probs = self.SEEDER_LIST[index](in_tensor).cpu().detach().numpy()
		out = np.argmax(out_probs, axis=1)
		if message : print("[INFO] Prediction : " + str(out))
		return np.squeeze(out)
	
	def train(self, output, target, generation=0, index=0, episod=0, i_batch=0, message=False):
		# reset
		if self.TYPE!="class":
			if message : print("[INFO] Switch to training mode..")
			self.SEEDER_LIST[index].train()
		if message : print("[INFO] Init gradient..")
		#self.optimizer[index].zero_grad()
		self.SEEDER_LIST[index].zero_grad()
		# correct timestep
		"""
		output = pack_padded_sequence(output, decode_lengths, batch_first=True)
		target = pack_padded_sequence(target, decode_lengths, batch_first=True)
		"""
		# loss computation
		loss = self.criterion[index](output, target)
		# do back-ward
		loss.backward()
		self.optimizer[index].step()
		# save loss
		self.loss = self.loss.append({'GEN':generation, 'IDX_SEED':index, 'EPISOD':episod, 'N_BATCH':i_batch, 'LOSS_VALUES':float(loss.cpu().detach().numpy())}, ignore_index=True)
	
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
				NET_S += [pRNN(GRAPH_S[-1].NEURON_LIST, self.BATCH, self.IO[0], self.DEVICE, STACK=self.TIME_DEP)]
			PARENT += [i+1]
		### mutation
		GRAPH_M = []
		NET_M = []
		for g,j in zip(GRAPH_S,GRAPH_IDX):
			for i in range(self.NB_EVOLUTION):
				GRAPH_M += [g.NEXT_GEN()]
				NET_M += [pRNN(GRAPH_M[-1].NEURON_LIST, self.BATCH, self.IO[0], self.DEVICE, STACK=self.TIME_DEP)]
				PARENT += [j+1]
		### news random
		GRAPH_N = []
		NET_N = []
		for n in range(self.NB_CHALLENGE):
			GRAPH_N += [GRAPH_EAT([self.IO, 1], None)]
			NET_N += [pRNN(GRAPH_N[-1].NEURON_LIST, self.BATCH, self.IO[0], self.DEVICE, STACK=self.TIME_DEP)]
			PARENT += [-1]
		### update seeder list and stock info
		self.PARENTING += [np.array(PARENT)[None]]
		self.GRAPH_LIST = GRAPH_S + GRAPH_M + GRAPH_N
		self.SEEDER_LIST = NET_C + NET_S + NET_M + NET_N
		### update model
		self.update_model()

	def fit(self, dataset):
		# SUPERVISED TRAIN
		for g in tqdm(range(self.NB_GEN)) :
			# Loading data for each gen
			data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.BATCH, shuffle=True)
			# Train
			for n in tqdm(range(self.NB_SEEDER)) :
				# switch torch model to train mode
				self.SEEDER_LIST[n].train()
				for batch_idx, (data, target) in enumerate(data_loader) :
					# vectorization
					x = data.reshape(self.BATCH,-1)
					# calculate
					output = self.SEEDER_LIST[n](x.to(self.DEVICE))
					# train
					self.train(output, target.to(self.DEVICE), g, n, 0, batch_idx)
					if batch_idx == self.NB_E_P_G :
						break
						# evaluate
						supp_factor = self.predict(data.to(self.DEVICE), n)
			# selection
			self.selection(g, supp_factor = 1)
		# finalization
		self.finalization()

	def finalization(self, supp_param=None, save=True):
		self.PARENTING = np.concatenate(self.PARENTING).T
		self.supp_param = supp_param
		if save :
			path = os.path.expanduser('~')+'/Saved_Model'
			if(not os.path.isdir(path)): os.makedirs(path)
			time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			filehandler = open(path+os.path.sep+"MODEL_FF_"+time+".obj", 'wb')
			pickle.dump(self, filehandler); filehandler.close()
	
	def load(self, Filename):
		with open('OUT'+os.path.sep+Filename, 'rb') as f:
			return pickle.load(f)

### basic exemple
if __name__ == '__main__' :
	import torchvision
	data_path = os.path.expanduser('~')+'/Dataset/MNIST'
	Transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((14,14)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
	mnist_dataset = torchvision.datasets.MNIST(data_path, download=True, transform=Transforms)
	## parameter
	IO =  (14*14,10)
	BATCH = 25
	NB_GEN = 3
	NB_SEED = 2**2
	#NB_EPISODE = 25000 #25000
	NB_EPOCH = 2
	TRAIN_LENGHT = int(mnist_dataset.train_data.shape[0] * NB_EPOCH)
	ALPHA = 0.9
	## load model
	model = FunctionalFilet([IO, BATCH, NB_GEN, NB_SEED, ALPHA, TRAIN_LENGHT], TYPE='class')
	## Train
	model.fit(mnist_dataset)
	## predict (note : linearize image !)
	X, Y = mnist_dataset.train_data, mnist_dataset.train_labels
	x, y = X[:BATCH,::2,::2].reshape(BATCH,-1), Y[:BATCH]
	y_pred = model.predict(x,0, True)
	y_pred = model.predict(x,1, True)
	print("[INFO] Labels is : "+str(y))

