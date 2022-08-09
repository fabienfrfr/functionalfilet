#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:40:35 2022
@author: fabien
"""

import gym
import torch, numpy as np, pandas as pd
from tqdm import tqdm
from collections import namedtuple

import FUNCTIONNAL_FILLET_ as FF

##################################### FUNCTION
## For Q-Learning
def Q_TABLE(model, batch, GAMMA = 0.9):
	old_state = torch.tensor(np.concatenate(batch.state), dtype=torch.float)
	action = torch.tensor(np.array(batch.action), dtype=torch.long).unsqueeze(1)
	new_state = torch.tensor(np.concatenate(batch.next_state), dtype=torch.float)
	reward = torch.tensor(np.array(batch.reward), dtype=torch.long)
	done = torch.tensor(np.array(batch.done), dtype=torch.int)
	# actor proba
	actor = model(old_state)
	# Compute predicted Q-values for each action
	pred_q_values_batch = actor.gather(1, action)
	pred_q_values_next  = model(new_state)
	# Compute targeted Q-value for action performed
	target_q_values_batch = (reward+(1-done)*GAMMA*torch.max(pred_q_values_next, 1)[0]).detach().unsqueeze(1)
	# return y, y_prev
	return pred_q_values_batch,target_q_values_batch

## Accuracy variable construction factor
def Factor_construction(df, main_group, main_value, sub_group, factor, AL=None):
	# extract data
	sub_df = df[df[main_group] == main_value]
	if AL == None :
		fact_abs = sub_df.groupby(sub_group)[factor].mean().values
	else :
		"""
		AL = ALPHA_LIST = [group, coeff]
		"""
		fact_abs = np.unique(sub_df[factor].values)
		for i,g in sub_df.groupby(sub_group)[factor] :
			Tail_eps = g[AL[0]].min()+(g[AL[0]].max() - g[AL[0]].min())*AL[1]
			fact_abs[int(i)] = g[g[AL[0]] > Tail_eps][factor].mean()
	fact_rel = (fact_abs-fact_abs.min())/(fact_abs.max()-fact_abs.min())
	return fact_rel

def FIT(env,MODEL):
	# loop
	dilemna_ = MODEL.NB_E_P_G*MODEL.ALPHA
	duration = pd.DataFrame(columns=['GEN','IDX_SEED', 'EPISOD', 'DURATION']).astype(int)
	for g in tqdm(range(MODEL.NB_GEN)) :
		# per seeder
		for n in range(MODEL.NB_SEEDER):
			# train
			render,l = 0,0
			for i in range(MODEL.NB_E_P_G):
				new_state = env.reset()
				done = False
				# gen data
				j = 0
				while not done :
					# train or predict (%)
					if n < dilemna_ :
						action = MODEL.STEP(new_state[None], n)
					else :
						action = MODEL.PREDICT(new_state[None], n)
					state = new_state
					new_state, reward, done, info = env.step(action)
					if done and j < N_MAX-10 :
						reward = -10
					# see
					if render == 0 :
						env.render()
					MODEL.memory[n].push(state[None], action, new_state[None], reward, done)
					# iteration
					j+=1
					l+=1
				# duration
				duration = duration.append({'GEN':g, 'IDX_SEED':n,'EPISOD':i,'DURATION':j},ignore_index=True)
				render+=1
				# fit
				if l >= MODEL.BATCH :
					nb_batch = min(int(MODEL.memory[n].__len__()/MODEL.BATCH), np.rint(j/MODEL.BATCH).astype(int))
					transitions = MODEL.memory[n].sample(nb_batch*MODEL.BATCH)
					# batch adapted loop
					for b in range(nb_batch) :
						batch = Transition(*zip(*transitions[b*MODEL.BATCH : (b+1)*MODEL.BATCH]))
						pred_q, target_q = Q_TABLE(MODEL.SEEDER_LIST[n], batch)
						MODEL.TRAIN(pred_q, target_q, g, n, i, b)
		# Accuracy contruction
		duration_factor = Factor_construction(duration, 'GEN', g, 'IDX_SEED', 'DURATION')
		duration_factor = 1 - duration_factor # for odering (loss need inversion)
		# Apply natural selection
		MODEL.SELECTION(g, supp_factor=duration_factor)
		#if g == 1 : break
	## Finalization
	MODEL.FINALIZATION(supp_param=duration,save=True)
	env.close()
	return duration

def RENDER(env,MODEL) :
	for n in range(MODEL.NB_SEEDER):
		new_state = env.reset()
		done = False
		while not done :
			action = MODEL.PREDICT(new_state[None], n)
			new_state, reward, done, info = env.step(action)
			env.render()

##################################### ALGO

if __name__ == '__main__' :
	LOAD = True #True
	Filename = 'MODEL_FF_20220125_172002.obj' #25 : 5h
	Filename = 'MODEL_FF_20220203_144846.obj' #25 : 5h
	## env gym
	env = gym.make("CartPole-v0")
	N_MAX = 300
	env._max_episode_steps=N_MAX
	NB_OBS = env.observation_space.shape[0]
	NB_ACTION = env.action_space.n
	
	## parameter
	IO =  (NB_OBS,NB_ACTION)
	BATCH = 25
	NB_GEN = 100
	NB_SEED = 5**2
	NB_EPISODE = 25000 #25000
	ALPHA = 0.9
	
	## Load previous model or launch new
	if LOAD :
		Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done')) #important
		MODEL = FF.FunctionnalFillet([IO, BATCH, NB_GEN, NB_SEED, NB_EPISODE, ALPHA], Transition, TYPE='RL')
		MODEL = MODEL.LOAD(Filename)
	else :
		Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))
		MODEL = FF.FunctionnalFillet([IO, BATCH, NB_GEN, NB_SEED, NB_EPISODE, ALPHA], Transition, TYPE='RL')
		# Fit
		FIT(env, MODEL)
	
	# visualization output training
	RENDER(env,MODEL)
	env.close()
	## extract some variable
	duration = MODEL.supp_param
	PARENT = MODEL.PARENTING.T
	NB_P_GEN = MODEL.NB_SEEDER
	
	### plot
	import pylab as plt
	from scipy.ndimage import filters
	
	plt.imshow(PARENT); plt.show(); plt.close()
	
	import EXTRA_FUNCTION as EF
	"""
	duration_sep = duration.groupby(['IDX_SEED','GEN']).agg({"DURATION":"mean"})
	plt.plot(duration_sep)
	"""
	## 2 simple curve
	ctrl = duration[duration.IDX_SEED==0].groupby('GEN')
	evol = duration[duration.IDX_SEED!=0].groupby('GEN')
	
	score, std, min_, max_ = [], [], [], []
	for c in [ctrl,evol] :
		score += [np.squeeze(c.agg({'DURATION':'max'}).values)] 
		std += [np.squeeze(c.agg({'DURATION':'std'}).values)] 
		min_ += [score[-1].min()]
		max_ += [score[-1].max()]
	min_, max_ = min(min_), max(max_)
	
	# norm & plot save
	curve, inter = [],[]
	for z in zip(score,std):
		curve += [filters.gaussian_filter1d((z[0]-min_)/(max_-min_),1)]
		inter += [filters.gaussian_filter1d((z[1])/(max_-min_),1)]	   
	EF.FAST_PLOT(curve, inter, ['ctrl','evolution'],'','','',yaxis = [0,1.5], BATCH=25,CYCLE=100, STD=1)
	
	## plot parenting
	N_AGENT_TOT = np.product(MODEL.PARENTING.shape)
	node = np.arange(N_AGENT_TOT)
	pos, G = EF.IMLINEAGE_2_GRAPH(node,PARENT)
	# calculate heritage
	G, edges_size, node_size, SHORT_PATH = EF.ADD_PATH(node,G)
	# prepare data indexes
	# show nodes
	N_ = np.sqrt(node_size.reshape((NB_GEN+1,NB_P_GEN)).T)
	YMAX = np.sqrt(N_.shape[1])
	# curve construc
	plt.imshow(N_, interpolation='none', aspect="auto"); plt.show(); plt.close()
	node = [N_[2:7].mean(0), N_[7:-3].mean(0), N_[-3:].mean(0)]
	std = [N_[2:7].std(0), N_[7:-3].std(0), N_[-3:].std(0)]
	# norm & plot
	curve, inter = [],[]
	for z in zip(node,std):
		curve += [filters.gaussian_filter1d(z[0],1)]
		inter += [filters.gaussian_filter1d(z[1],1)]	   
	EF.FAST_PLOT(curve, inter, ['parent','child','random'],'node CartPole-v0','NODES','GEN',yaxis = [1,4], BATCH=25,CYCLE=100, STD=1)