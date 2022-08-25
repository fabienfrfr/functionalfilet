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
from torch.utils.data import TensorDataset

# system module
import pickle, datetime, json
import os, time, copy, ast
from tqdm import tqdm
import multiprocessing as mp

# networks construction
from functionalfilet.ENN_net import EvoNeuralNet

# utils
from functionalfilet.utils import ReplayMemory, F1_Score

##### FF MODULE
"""  
Note hybrid propriety :   
If GEN = 0, equivalent of no evolution during training : only SGD
if NB_BATCH > NB_BATCH/GEN, equivalent of no SGD : only evolution
"""
class FunctionalFilet():
	def __init__(self, io=(64,16), batch=25, nb_gen=100, nb_seed=9, alpha=0.9, train_size=1e6, NAMED_MEMORY=None, TYPE="class", INVERT=False, DEVICE=True, TIME_DEPENDANT = False, GDchain="standard", lossF = "standard", metrics='standard', multiprocessing=False):
		print("[INFO] Starting System...")
		# parameter
		self.IO =  io
		if INVERT==True :
			# Feature augmentation (ex : after bottleneck)
			self.IO = tuple(reversed(self.IO))
		elif INVERT=="same":
			# f: R -> R
			io = int(np.sqrt(np.prod(io))) # geometric mean
			self.IO = (io,io)
		self.BATCH = batch
		self.NB_GEN = nb_gen
		self.NB_SEEDER = max(4,int(np.rint(np.sqrt(nb_seed))**2))
		self.ALPHA = alpha # 1-% of predict (not random step)
		if TYPE == "class" or TYPE == "regress" :
			self.NB_BATCH = int(train_size / self.BATCH)  # nb_batch = (dataset_lenght * nb_epoch) / batch_size
			self.NB_EPISOD = self.NB_BATCH # only in supervised learning
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
		self.RealIO = None
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
		self.GDchain, self.lossF, self.metrics = GDchain, lossF, metrics
		self.optimizer, self.criterion = None, None
		self.update_model()
		print("[INFO] Generate evolution variable for population..")
		# selection
		self.loss = pd.DataFrame(columns=['GEN','IDX_SEED', 'EPISOD', 'N_BATCH', 'LOSS_VALUES'])
		self.test = pd.DataFrame(columns=['GEN', 'IDX_SEED', 'SCORE', 'TRUE','PRED'])
		# evolution variable
		self.PARENTING = [-1*np.ones(self.NB_SEEDER)[None]]
		self.PARENTING[0][0][:self.NB_CONTROL] = 0
		# checkpoint & process
		self.checkpoint = []
		self.multiprocessing = multiprocessing
		self.cpuCount = int(os.cpu_count()/2)
		print("[INFO] Model pre-created!")

	def set_IO(self, x):
		self.RealIO = x

	def update_model(self):
		# refresh memory
		del self.optimizer
		del self.criterion
		## Gradient descent (chain rule) algo
		if self.GDchain == 'standard' :
			self.optimizer = [torch.optim.Adam(s.parameters()) for s in self.SEEDER_LIST]
		else :
			# custom sgd algorithm
			self.optimizer = [self.GDchain(s.parameters()) for s in self.SEEDER_LIST]
		## Error function
		if self.lossF == 'standard' :
			if self.TYPE == "class" :
				self.criterion = [nn.CrossEntropyLoss().to(self.DEVICE) for n in range(self.NB_SEEDER)]
			else :
				self.criterion = [nn.SmoothL1Loss().to(self.DEVICE) for n in range(self.NB_SEEDER)] # regression / RL
		else :
			# custom loss function
			self.criterion = [self.lossF().to(self.DEVICE) for n in range(self.NB_SEEDER)]
		# memory (unused in supervised learning)
		if self.NAMED_M == None :
			self.memory = {"X_train":None, "Y_train":None, "X_test":None, "Y_test":None}
		else :
			self.memory = [ReplayMemory(1024, self.NAMED_M) for n in range(self.NB_SEEDER)]

	# Particular case : RL
	def step(self, INPUT, index=0, message=False):
		in_tensor = self.to_tensor(INPUT).type(torch.float)
		# device
		in_tensor = in_tensor.to(self.DEVICE)
		# linearize
		shape = tuple(in_tensor.shape)
		in_tensor = in_tensor.reshape(shape[0],-1)
		if message : print("[INFO] Switch to inference mode for model:"+str(index))
		# activate all link
		self.SEEDER_LIST[index].eval()
		out_probs = self.SEEDER_LIST[index](in_tensor)
		# exploration dilemna
		DILEMNA = np.squeeze(out_probs.cpu().detach().numpy())
		if DILEMNA.sum() == 0 or str(DILEMNA.sum()) == 'nan' :
			out_choice = np.random.randint(self.RealIO[1])
		else :
			if DILEMNA.min() < 0 : DILEMNA = DILEMNA-DILEMNA.min() # order garanty
			## ADD dispersion between near values (ex : q-table, values is near)
			order = np.argsort(DILEMNA)+1
			#order[np.argmax(order)] += 1
			order = np.exp(order)
			# probability
			p_norm = order/order.sum()
			out_choice = np.random.choice(self.RealIO[1], p=p_norm)
		if message : print("[INFO] Chosen prediction : " + str(out_choice))
		return out_choice
	
	def predict(self, INPUT, index=0, message=True, numpy=False, argmax=False):
		in_tensor = self.to_tensor(INPUT).type(torch.float)
		# device
		in_tensor = in_tensor.to(self.DEVICE)
		# linearize
		shape = tuple(in_tensor.shape)
		in_tensor = in_tensor.reshape(shape[0],-1)
		# extract prob
		if message : print("[INFO] Switch to inference mode for model:"+str(index))
		self.SEEDER_LIST[index].eval()
		if message and (shape[0]>self.BATCH) : 
			print("[INFO] Input size dimension batch it's superior of model capacity")
		if (shape[0]<= self.BATCH) :
			out_probs = self.SEEDER_LIST[index](in_tensor)
		else :
			in_tensor = torch.split(in_tensor, self.BATCH)
			out_probs = torch.cat([self.SEEDER_LIST[index](i) for i in in_tensor])
		out = out_probs #.squeeze()
		if argmax :
			out = torch.argmax(out, dim=1)
		if numpy :
			return out.cpu().detach().numpy()
		return out

	def train(self, output, target, generation=0, index=0, episod=0, i_batch=0, message=False):
		# reset
		if self.TYPE!="class":
			if message : print("[INFO] Switch to training mode..")
			self.SEEDER_LIST[index].train()
		if message : print("[INFO] Init gradient..")
		self.optimizer[index].zero_grad() 
		#self.SEEDER_LIST[index].zero_grad() # equiv
		# correct timestep (not included for now)
		if self.TIME_DEP :
			#output = pack_padded_sequence(output, decode_lengths, batch_first=True)
			#target = pack_padded_sequence(target, decode_lengths, batch_first=True)
			pass
		# loss computation
		loss = self.criterion[index](output, target)
		# do back-ward
		loss.backward()
		self.optimizer[index].step()
		# save loss
		self.loss = self.loss.append({'GEN':generation, 'IDX_SEED':int(index), 'EPISOD':episod, 'N_BATCH':i_batch, 'LOSS_VALUES':float(loss.cpu().detach().numpy())}, ignore_index=True)
	
	def add_checkpoint(self, gen):
		i = 0
		for s in self.SEEDER_LIST[:self.NB_CONTROL+self.NB_EVOLUTION] :
			self.checkpoint += [{'GEN':gen,'IDX_SEED':int(i), 'CONTROL':s.control, 'GRAPH':s.net, 'LIST_C': s.graph.LIST_C, 'NETWORKS': s.state_dict()}]
			i+=1

	def selection(self, GEN):
		# sup median loss selection
		TailLoss = np.ones(self.NB_SEEDER)
		supp_factor = np.ones(self.NB_SEEDER)
		# extract data
		sub_loss = self.loss[self.loss.GEN == GEN]
		sub_test = self.test[self.test.GEN == GEN]
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
			# loss normalization
			relativeLOSS = (TailLoss-TailLoss.min())/(TailLoss.max()-TailLoss.min())
			# score metric normalization (if necessary)
			relativeSCOR = (sub_test.SCORE - sub_test.SCORE.min()) / (sub_test.SCORE.max() - sub_test.SCORE.min())
			supp_factor[list(sub_test.IDX_SEED.astype(int).values)] = relativeSCOR.values
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
			net.checkIO(*self.RealIO)
			NET_N += [net]
			PARENT += [-1]
		### update seeder list, refresh memory and stock info
		self.PARENTING += [np.array(PARENT)[None]]
		del self.SEEDER_LIST
		self.SEEDER_LIST = NET_C + NET_S + NET_M + NET_N
		### update model 
		self.update_model()
		if self.DEVICE.type == 'cuda':
			torch.cuda.empty_cache()

	def process(self, g, n, data_loader) :
		# switch torch model to train mode
		self.SEEDER_LIST[n].train()
		for batch_idx, (data, target) in enumerate(data_loader) :
			data, target = data.to(self.DEVICE, non_blocking=True), target.to(self.DEVICE, non_blocking=True)
			# 1D vectorization
			x = data.reshape(self.BATCH,-1)
			y_ = target.reshape(self.BATCH,-1).squeeze()
			# calculate
			output = self.SEEDER_LIST[n](x)
			# train (note : in supervised learning, episode = batch_idx)
			self.train(output, y_, g, n, batch_idx, batch_idx)
			if batch_idx == self.NB_E_P_G - 1 :
				# evaluate test error
				out = self.predict(data, n, message=False)
				if self.metrics == "standard" :
					if self.TYPE == "regress" :
						R2Score = lambda y,y_pred: 1 - torch.sum((y-y_pred)**2)/torch.sum((y-torch.mean(y))**2)
						score = R2Score(y_, out)
					elif self.TYPE == "class" :
						# adapted for unbalanced data
						score = F1_Score(y_, out)
				# custom metrics
				else :
					score = self.metrics(y_, out)
				# save loss (adding input)
				self.test = self.test.append({'GEN':g, 'IDX_SEED':int(n), 'SCORE':float(score.cpu().detach().numpy()), 'TRUE':y_.cpu().detach().numpy().tolist(), 'PRED':out.cpu().detach().numpy().tolist()}, ignore_index=True)
				# BREAK DATA LOADER LOOP
				break

	def to_tensor(self, X):
		if isinstance(X, torch.Tensor) :
			return X
		else :
			return torch.tensor(X)

	def fit(self, X, y, sample_weight=None):
		### SUPERVISED TRAIN
		X,y = self.to_tensor(X), self.to_tensor(y)
		# data real shape
		shape = tuple(X.shape)
		nb_sample, input_size = shape[0], np.prod(shape[1:])
		if self.TYPE == "class" :
			output_size = y.unique().size()[0]
		else :
			output_size = torch.prod(torch.tensor(y.shape)[1:]).item()
		self.RealIO = input_size, output_size
		print("[INFO] Your dataset I/O is : " + str((nb_sample,self.RealIO)))
		print("[WARNING] If you have a 2D output (ex: Image), you need to reshape output prediction !")
		# data verification size
		data_ratio = nb_sample / self.BATCH
		if self.NB_E_P_G > data_ratio :
			augment = int(self.NB_E_P_G/data_ratio)
			print("[INFO] Your dataset contains less than min sample per generation. Data augmentation is : " +str(augment))
			X = torch.cat([X for _i in range(2*augment+1)])
			y = torch.cat([y for _i in range(2*augment+1)])
		# adjust I/O
		print("[INFO] Apply IO modification if undefined..")
		for s in self.SEEDER_LIST : s.checkIO(*self.RealIO)
		print("[INFO] Contruct Dataset formats..")
		dataset = TensorDataset(X,y)
		# generation loop
		print("[INFO] Launch evolution algorithm with SGD !")
		for g in tqdm(range(self.NB_GEN)) :
			# Loading data for each gen
			data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.BATCH, shuffle=True, num_workers=self.cpuCount)
			# Train 
			if self.multiprocessing :
				process = []
				# adapt to number of cpu (to code)
				for rank in range(self.NB_SEEDER) :
					p = mp.Process(target=self.process, args=(g,rank,data_loader,))
					p.start()
					process.append(p)
				for p in process :
					p.join()
			else :
				for n in range(self.NB_SEEDER) :
					self.process(g, n, data_loader)
			# save data
			self.add_checkpoint(g)
			# selection
			self.selection(g)
		# finalization
		self.finalization()

	def finalization(self, save=True):
		self.PARENTING = np.concatenate(self.PARENTING).T
		if save :
			print("[INFO] Save model...")
			time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			path = os.path.expanduser('~')+'/Saved_Model/ff_'+ self.TYPE + '_' + time
			if(not os.path.isdir(path)): os.makedirs(path)
			## model checkpoint
			df = pd.DataFrame(self.checkpoint)
			df.to_pickle(path+os.path.sep+"saved_ffcheckpoints_.obj") # compressed
			self.checkpoint = df
			## model score
			self.loss.to_csv(path+os.path.sep+"score_loss_ff.csv")
			self.test.to_csv(path+os.path.sep+"score_test_ff.csv")
			## model evolution
			np.save(path+os.path.sep+"phylogenicTree.npy", self.PARENTING)
			## model param
			param = { 'io':self.IO,'batch':self.BATCH,'nb_gen':self.NB_GEN,'nb_seed':self.NB_SEEDER,'alpha':self.ALPHA,'trainsize':self.NB_EPISOD,'type':self.TYPE,'timedep':self.TIME_DEP,'device':self.DEVICE.type,'invert':self.INVERT,'realIO':self.RealIO,'SGD':self.GDchain,'LOSSF':self.lossF,'METRICS':self.metrics}
			param = {k:str(v) for k,v in param.items()}
			with open(path+os.path.sep+"model_parameter.json", "w") as outfile:
				json.dump(param, outfile)
			print("[INFO] The model its saved !")
	
	def load(self, folder):
		print("[INFO] You chose to load model : " + folder)
		list_file = os.listdir(folder)
		# extract path
		checkpoint_path = folder + os.path.sep + list_file[np.argmax(["checkpoints" in f for f in list_file])]
		loss_path = folder + os.path.sep + list_file[np.argmax(["loss" in f for f in list_file])]
		test_path = folder + os.path.sep + list_file[np.argmax(["test" in f for f in list_file])]
		evolution_path = folder + os.path.sep + list_file[np.argmax(["phylogenic" in f for f in list_file])]
		param_path = folder + os.path.sep + list_file[np.argmax(["parameter" in f for f in list_file])]
		# basic parameter :
		with open(param_path) as f :
			self.parameter = json.load(f)
		self.IO = ast.literal_eval(self.parameter['io'])
		self.RealIO = ast.literal_eval(self.parameter['realIO'])
		self.BATCH = ast.literal_eval(self.parameter['batch'])
		# evolution
		self.PARENTING = np.load(evolution_path)
		# score
		self.loss = pd.read_csv(loss_path)
		with open(test_path) as f: col_names = f.readline().split('\n')[0].split(';')
		col_type = {col : ast.literal_eval for col in col_names}
		self.test = pd.read_csv(test_path)
		# checkpoint
		self.checkpoint = pd.read_pickle(checkpoint_path)
		## seeder
		last = self.checkpoint[self.checkpoint.GEN == self.checkpoint.GEN.max()]
		self.SEEDER_LIST = []
		for idx, ckpt in last.iterrows():
			self.SEEDER_LIST += [EvoNeuralNet(self.IO, self.BATCH, self.DEVICE, control=ckpt.CONTROL, graph=True, net=(self.IO, ckpt.GRAPH, ckpt.LIST_C))]
			self.SEEDER_LIST[-1].checkIO(*self.RealIO)
			self.SEEDER_LIST[-1].load_state_dict(ckpt.NETWORKS)
		print("[INFO] The model its loaded !")
