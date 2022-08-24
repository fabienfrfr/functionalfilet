# 20220810 fabienfrfr
import numpy as np

import torch, torch.nn as nn
import torch.nn.functional as F

from collections import deque
import itertools

class CTRL_NET(nn.Module):
	def __init__(self, IO, device):
		super(CTRL_NET, self).__init__()
		I,O = IO
		if I+O > 64 :
			H = 2*int(np.sqrt(I+O))
		else : 
			H = 16
		self.IN = nn.Conv1d(I, I, 1, groups=I, bias=True).to(device)
		self.H1 = nn.Linear(I, H).to(device)
		self.H2 = nn.Linear(H, H).to(device)
		self.OUT = nn.Linear(H, O).to(device)
		# net
		self.net = np.array([[I,H,H,O]])

	def forward(self, x):
		s = x.shape
		x = self.IN(x.view(s[0],s[1],1)).view(s)
		x = F.relu(x)
		x = F.relu(self.H1(x))
		x = F.relu(self.H2(x))
		return self.OUT(x)

# Label 1D long Tensor (without grad)
def one_hot(Tensor_, num_classes=-1):
	if num_classes == -1 :
		L = int(Tensor_.max()+1)
	else : 
		L = num_classes
	N = Tensor_.shape[0]
	# create new tensor
	Tensor = torch.zeros((N,L)).to(Tensor_.device)
	idx = torch.cat([torch.arange(N).to(Tensor_.device)[None], Tensor_[None]])
	Tensor[tuple(map(tuple,idx))] = 1
	return Tensor

def F1_Score(label, y_pred, epsilon=1e-7):
	# adapted of SuperShinyEyes code
	y_true = one_hot(label, num_classes=y_pred.shape[1]).to(torch.float32)
	y_pred = F.softmax(y_pred, dim=1).detach()
	# confusion matrix
	tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
	tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
	fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
	fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
	# precision, recall
	precision = tp / (tp + fp + epsilon)
	recall = tp / (tp + fn + epsilon)
	# f1
	f1 = 2* (precision*recall) / (precision + recall + epsilon)
	f1 = f1.clamp(min=epsilon, max=1-epsilon)
	return 1 - f1.mean()

## For Q-Learning
def Q_TABLE(NNmodel, batch, GAMMA = 0.9,  DEVICE=torch.device('cpu')):
	old_state = torch.tensor(np.concatenate(batch.state), dtype=torch.float).to(DEVICE)
	action = torch.tensor(np.array(batch.action), dtype=torch.long).unsqueeze(1).to(DEVICE)
	new_state = torch.tensor(np.concatenate(batch.next_state), dtype=torch.float).to(DEVICE)
	reward = torch.tensor(np.array(batch.reward), dtype=torch.float).to(DEVICE)
	done = torch.tensor(np.array(batch.done).astype(int), dtype=torch.float).to(DEVICE)
	# actor proba
	actor = NNmodel(old_state)
	# Compute predicted Q-values for each action
	pred_q_values_batch = actor.gather(1, action)
	pred_q_values_next  = NNmodel(new_state)
	# Compute targeted Q-value for action performed
	target_q_values_batch = (reward+(1-done)*GAMMA*torch.max(pred_q_values_next, 1)[0]).detach().unsqueeze(1)
	# return y, y_prev
	return pred_q_values_batch,target_q_values_batch

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