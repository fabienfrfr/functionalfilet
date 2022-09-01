# fabienfrfr 20220819
"""
	Reinforcement Q-learning

This very simple case makes it possible to understand the use of the model in the case 
of a markov decision process.

In this example, the model is adapted for unsupervised learning. It's highlights use
decomposition model (step per step), time dependance and dilemna problematic's in evolution case.

"""

## package
import os, numpy as np
import torch, gym
from collections import namedtuple

from tqdm import tqdm

## model import
from functionalfilet import model as ff, utils as f

## ff model
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))
model = ff.FunctionalFilet(train_size=1e3, TYPE="RL", NAMED_MEMORY=Transition, TIME_DEPENDANT = True)

## RL environment
print("[INFO] Launch reinforcement learning environment..")
env = gym.make("CartPole-v0")
model.set_IO((env.observation_space.shape[0], env.action_space.n))

print("[INFO] Apply IO modification..")
for s in model.SEEDER_LIST : s.checkIO(*model.RealIO)

load_name = 'RL_20220824_162415'
path = os.path.expanduser('~')+'/Saved_Model/ff_' + load_name
if os.path.isdir(path):
	### LOAD
	model.load(path)
else :
	## Manual fit
	print("[INFO] Start training..")
	dilemna_ = model.NB_E_P_G * model.ALPHA #apply dilemna for alpha% timestep
	for g in tqdm(range(model.NB_GEN)):
		# for each seeder
		for n in range(model.NB_SEEDER):
			# train
			render, lenght = 0,0
			for i in range(model.NB_E_P_G) :
				new_state = env.reset()
				done = False
				# generate data
				duration = 0
				while not done :
					# dilemna step or direct predict
					if i < dilemna_ :
						action = model.step(new_state[None], n)
					else :
						action = model.predict(new_state[None], n, message=False, numpy=True, argmax=True)[0]
					state = new_state
					# t+1
					new_state, reward, done, info = env.step(action)
					if done and duration < env._max_episode_steps - 0.05*env._max_episode_steps :
						reward = -10
					# show first episod training
					if render == 0 :
						env.render()
					# save step
					model.memory[n].push(state[None], action, new_state[None], reward, done)
					# iterate
					duration += 1
					lenght += 1
				# duration score
				if i == model.NB_E_P_G - 1 :
					model.test = model.test.append({'GEN':g, 'IDX_SEED':int(n),'SCORE':duration, 'TRUE':None,'PRED':None},ignore_index=True)
				render += 1
				# train step 
				if lenght >= model.BATCH :
					nb_batch = max(1,min(int(model.memory[n].__len__()/model.BATCH), np.rint(duration/model.BATCH).astype(int)))
					transitions = model.memory[n].sample(int(nb_batch*model.BATCH))
					# batch adapted loop
					for b in range(nb_batch) :
						batch = Transition(*zip(*transitions[b*model.BATCH : (b+1)*model.BATCH]))
						pred_q, target_q = f.Q_TABLE(model.SEEDER_LIST[n], batch, DEVICE=model.DEVICE)
						model.train(pred_q, target_q, g, n, i, b)
		# Add checkpoint
		model.add_checkpoint(g)
		# Apply natural selection
		model.selection(g)
	# Finalization
	model.finalization()

## Testing
for n in range(len(model.SEEDER_LIST)):
	print("[INFO] Render prediction of seeder : " +str(n))
	new_state = env.reset()
	done = False
	while not done :
		action = model.predict(new_state[None], n, message=False, numpy=True, argmax=True)[0]
		new_state, reward, done, info = env.step(action)
		env.render()

## Close environment
env.close()