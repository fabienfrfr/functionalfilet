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
import os, time, copy
from tqdm import tqdm

# networks construction
from functionalfilet.ENN_net import EvoNeuralNet

# utils
from functionalfilet.utils import ReplayMemory

##### FF MODULE
"""  
Note hybrid propriety :   
If GEN = 0, equivalent of no evolution during training : only SGD
if NB_BATCH > NB_BATCH/GEN, equivalent of no SGD : only evolution
"""
class FunctionalFilet():
	def __init__(self, io=(64,16), batch=25, nb_gen=100, nb_seed=9, alpha=0.9, train_size=1e6, NAMED_MEMORY=None, TYPE="class", INVERT=False, DEVICE=True, TIME_DEPENDANT = False):
		print("[INFO] Starting System...")
		# parameter
		self.IO =  io
		if INVERT :
			# Feature augmentation (ex : after bottleneck)
			self.IO = tuple(reversed(self.IO))
		self.BATCH = batch
		self.NB_GEN = nb_gen
		self.NB_SEEDER = max(4,int(np.rint(np.sqrt(nb_seed))**2))
		self.ALPHA = alpha # 1-% of predict (not random step)
		if TYPE == "class" :
			self.NB_BATCH = int(train_size / self.BATCH)  # nb_batch = (dataset_lenght * nb_epoch) / batch_size
			self.NB_E_P_G = int(self.NB_BATCH/self.NB_GEN)
		else :
			self.NB_EPISOD = train_size
			self.NB_E_P_G = int(self.NB_EPISOD/self.NB_GEN)
		self.TIME_DEP = TIME_DEPENDANT
		self.TYPE = TYPE
		self.NAMED_M = NAMED_MEMORY
		if DEVICE==True :
			self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		else :
			self.DEVICE = DEVICE
		self.INVERT = INVERT
		print("[INFO] Calculation type : " + self.DEVICE.type)
		print("[INFO] Generate selection parameters for population..")
		# evolution param
		self.NB_CONTROL = int(np.power(self.NB_SEEDER, 1./4))
		self.NB_EVOLUTION = int(np.sqrt(self.NB_SEEDER)-1) # square completion
		self.NB_CHALLENGE = int(self.NB_SEEDER - (self.NB_EVOLUTION*(self.NB_EVOLUTION+1) + self.NB_CONTROL))
		print("[INFO] Generate first evolutionnal neural networks..")
		self.SEEDER_LIST = [EvoNeuralNet(self.IO, self.BATCH, self.DEVICE, control=True) for n in range(self.NB_CONTROL)]
		for _n in range(self.NB_SEEDER-self.NB_CONTROL) :
			self.SEEDER_LIST += [EvoNeuralNet(self.IO, self.BATCH, self.DEVICE, stack=self.TIME_DEP)]
		print("[INFO] ENN Generated!")
		# training parameter
		print("[INFO] Generate training parameters for population..")
		self.update_model()
		print("[INFO] Generate evolution variable for population..")
		# selection
		self.loss = pd.DataFrame(columns=['GEN','IDX_SEED', 'EPISOD', 'N_BATCH', 'LOSS_VALUES'])
		self.supp_param = None
		# evolution variable
		self.PARENTING = [-1*np.ones(self.NB_SEEDER)[None]]
		self.PARENTING[0][0][:self.NB_CONTROL] = 0
		# checkpoint
		self.checkpoint = []
		print("[INFO] Model created!")
		
	def update_model(self):
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
		# activate all link
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
		in_tensor = in_tensor.to(self.DEVICE)
		# extract prob
		if message : print("[INFO] Switch to inference mode for model:"+str(index))
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
		self.optimizer[index].zero_grad() # better if multiple block ?
		#self.SEEDER_LIST[index].zero_grad()
		# correct timestep (not included)
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
	
	def add_checkpoint(self, gen):
		i = 0
		for s in self.SEEDER_LIST :
			self.checkpoint += [{'GEN':gen,'IDX_SEED':i, 'GRAPH':s.net, 'NETWORKS': s.state_dict()}]
			i+=1

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
		NET_S = []
		GRAPH_IDX = list(order[:self.NB_EVOLUTION])
		for i in GRAPH_IDX :
			if np.random.choice((False,True), 1, p=[1./self.NB_GEN,1-1./self.NB_GEN]):
				NET_S += [self.SEEDER_LIST[self.NB_CONTROL:][i].update('copy')]
			else :
				NET_S += [self.SEEDER_LIST[self.NB_CONTROL:][i].update('reset')]
			PARENT += [i+1]
		### mutation
		NET_M = []
		for j in GRAPH_IDX:
			for i in range(self.NB_EVOLUTION):
				NET_M += [self.SEEDER_LIST[self.NB_CONTROL:][i].update('mut')]
				PARENT += [j+1]
		### news random
		NET_N = []
		for n in range(self.NB_CHALLENGE):
			net = EvoNeuralNet(self.IO, self.BATCH, self.DEVICE)
			net.checkIO(self.RI, self.RO)
			NET_N += [net]
			PARENT += [-1]
		### update seeder list and stock info
		self.PARENTING += [np.array(PARENT)[None]]
		self.SEEDER_LIST = NET_C + NET_S + NET_M + NET_N
		### update model
		self.update_model()

	def fit(self, dataset):
		### SUPERVISED TRAIN
		s_I, self.RO = dataset.train_data.shape, dataset.train_labels.unique().size()[0]
		self.RI = np.prod(s_I[1:])
		# adjust I/O
		for s in self.SEEDER_LIST : s.checkIO(self.RI, self.RO)
		# generation loop
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
			self.add_checkpoint(g)
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
			## model checkpoint
			df = pd.DataFrame(self.checkpoint)
			df.to_pickle(path+os.path.sep+"saved_ffcheckpoints_"+time+".obj") # compressed
			"""
			filehandler = open(path+os.path.sep+"MODEL_FF_"+time+".obj", 'wb')
			pickle.dump(self, filehandler); filehandler.close()
			"""
			## model score
			self.loss.to_csv(path+os.path.sep+"score_loss_ff"+time+".csv")
			## model evolution
			#self.PARENTING
			## model param
			param = {}
			#param = pd.DataFrame(param).to_csv("")
	
	def load(self, Date_filename):
		#target_net.load_state_dict(policy_net.state_dict())
		with open('OUT'+os.path.sep+Filename, 'rb') as f:
			return pickle.load(f)

### basic exemple
if __name__ == '__main__' :
	import torchvision
	data_path = os.path.expanduser('~')+'/Dataset/MNIST'
	Transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]) # torchvision.transforms.Resize((14,14))
	mnist_dataset = torchvision.datasets.MNIST(data_path, download=True, transform=Transforms)
	## parameter (not default)
	NB_EPOCH = 2
	TRAIN_LENGHT = int(mnist_dataset.train_data.shape[0] * NB_EPOCH)/25
	## load model
	model = FunctionalFilet(nb_seed=4, train_size=TRAIN_LENGHT, TYPE='class')
	## Train
	model.fit(mnist_dataset)
	## predict (note : linearize image !)
	BATCH = model.BATCH
	"""
	X, Y = mnist_dataset.train_data, mnist_dataset.train_labels
	x, y = X[:BATCH,::2,::2].reshape(BATCH,-1), Y[:BATCH]
	y_pred = model.predict(x,0, True)
	y_pred = model.predict(x,1, True)
	print("[INFO] Labels is : "+str(y))
	"""
