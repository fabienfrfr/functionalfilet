#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:38:14 2021
@author: fabien
"""
import torch, torch.nn as nn
import numpy as np, pylab as plt
from GRAPH_EAT import GRAPH_EAT
from pRNN_GEN import pRNN
import torch.nn.functional as F

# control net
class CTRL_NET(nn.Module):
    def __init__(self, IO):
        super(CTRL_NET, self).__init__()
        I,O = IO
        H = np.rint(np.sqrt(I+O)).astype(int)
        self.IN = nn.Conv1d(I, I, 1, groups=I, bias=True)
        self.H1 = nn.Linear(I, H)
        self.H2 = nn.Linear(H, H)
        self.OUT = nn.Linear(H, O)

    def forward(self, x):
        s = x.shape
        x = F.relu(self.IN(x.view(s[0],s[1],1)).view(s))
        x = F.relu(self.H1(x))
        x = F.relu(self.H2(x))
        return self.OUT(x)

################################ AGENT
class Q_AGENT():
    def __init__(self, *arg, MODEL = None, CTRL=False, NET = None, COOR = None):
        self.P_MIN = 1
        # Parameter
        self.ARG = arg
        self.IO = arg[0] # image cells, action
        self.NB_P_GEN = arg[1]
        self.batch_size = arg[2]
        self.N_TIME = arg[4]
        self.N_CYCLE = arg[5]
        ## Init
        if CTRL :
            self.NET = GRAPH_EAT(None, self.CONTROL_NETWORK(self.IO))
            MODEL = CTRL_NET(self.IO)
        elif NET == None :
            self.NET = GRAPH_EAT([self.IO, self.P_MIN], None)
        else :
            self.NET = NET
        self.NEURON_LIST = self.NET.NEURON_LIST
        if (MODEL == None) and not(CTRL) :
            self.MODEL = pRNN(self.NEURON_LIST, self.batch_size, self.IO[0], STACK=True)
        else :
            self.MODEL = MODEL
        # nn optimiser
        self.GAMMA = 0.9
        #self.optimizer = torch.optim.Adam(self.MODEL.parameters())
        self.optimizer = torch.optim.Adam(self.MODEL.parameters()) #torch.optim.SGD(self.MODEL.parameters(), lr=1e-6, momentum=0.9)
        self.criterion = nn.SmoothL1Loss() # HubberLoss, nn.MSELoss() # because not classification (same comparaison [batch] -> [batch])
        #self.criterion = nn.NLLLoss(reduction='sum') #negative log likelihood loss ([batch,Nout]->[batch])
        self.loss = None
        self.LOSS = []

        ## IO Coordinate
        X_A = np.mgrid[-1:2,-1:2].reshape((2,-1)).T
        X_B = np.array([[0,0],[0,2],[0,4],[2,0],[2,4],[4,0],[4,2],[4,4]])-[2,2]
        self.X,self.Y = np.concatenate((X_A,X_B)), np.array([[0,1],[1,2],[2,0]])-[1,1]
        ## Data sample (memory : 'old_state', 'action', 'new_state', 'reward', 'terminal')
        self.MEMORY = [[],[],[],[],[]]
        self.MEMORY_ = None
        ## Players
        self.prev_state = None
    
    def INIT_ENV(self, ENV_INPUT) :
        self.prev_state = ENV_INPUT.FIRST_STEP_SET(0)

    def PARTY(self, ENV_INPUT):
        # reset game environement (for each batch ? important) 
        for n in range(self.N_CYCLE):
            self.prev_state = ENV_INPUT.FIRST_STEP_SET(n)
            for t in range(self.N_TIME):
                # loop game
                for i in range(self.batch_size):
                    action = self.ACTION(self.prev_state)
                    new_state, reward, DONE = ENV_INPUT.STEP(action)
                    # Memory update
                    if i == self.batch_size-1 : DONE = True
                    self.SEQUENCING(self.prev_state,action,new_state,reward,DONE)
                    # n+1
                    self.prev_state = new_state.copy()
                    # escape loop
                    if DONE == True : break                
                # Reinforcement learning
                self.OPTIM()

    ## Action Exploration/Exploitation Dilemna
    def ACTION(self, Input) :
        img_in = torch.tensor(Input, dtype=torch.float)
        # actor-critic (old version)
        action_probs = self.MODEL(img_in)
        # exploration-exploitation dilemna
        DILEMNA = np.squeeze(action_probs.detach().numpy())
        if DILEMNA.sum() == 0 or str(DILEMNA.sum()) == 'nan' :
            next_action = np.random.randint(self.IO[1])
        else :
            if DILEMNA.min() < 0 : DILEMNA = DILEMNA-DILEMNA.min() # n-1 choice restriction
            ## add dispersion (in q-table, values is near)
            order = np.exp(np.argsort(DILEMNA)+1)
            # probability
            p_norm = order/order.sum()
            #print(order, p_norm)
            next_action = np.random.choice(self.IO[1], p=p_norm)
        return next_action
    
    ## Memory sequencing
    def SEQUENCING(self, prev_state,action,new_state,reward,DONE):
        self.MEMORY[0] += [prev_state]
        self.MEMORY[1] += [action]
        self.MEMORY[2] += [new_state]
        self.MEMORY[3] += [reward]
        self.MEMORY[4] += [DONE]
        if DONE :
            self.MEMORY[0] = torch.tensor(np.concatenate(self.MEMORY[0]), dtype=torch.float)
            self.MEMORY[1] = torch.tensor(np.array(self.MEMORY[1]),  dtype=torch.long).unsqueeze(1)
            self.MEMORY[2] = torch.tensor(np.concatenate(self.MEMORY[2]), dtype=torch.float)
            self.MEMORY[3] = torch.tensor(np.array(self.MEMORY[3]))
            self.MEMORY[4] = torch.tensor(np.array(self.MEMORY[4]), dtype=torch.int)
    
    ## Training Q-Table
    def OPTIM(self) :
        # extract info
        old_state, action, new_state, reward, DONE = self.MEMORY
        # actor proba
        actor = self.MODEL(old_state)
        # Compute predicted Q-values for each action
        pred_q_values_batch = actor.gather(1, action)
        pred_q_values_next  = self.MODEL(new_state)
        # Compute targeted Q-value for action performed
        target_q_values_batch = (reward+(1-DONE)*self.GAMMA*torch.max(pred_q_values_next, 1)[0]).detach().unsqueeze(1)
        self.y = [pred_q_values_batch,target_q_values_batch]
        #[print(i,self.y[i].shape) for i in range(2)]
        #print(self.y[1])
        # zero the parameter gradients
        self.MODEL.zero_grad()
        # Compute the loss
        self.loss = self.criterion(pred_q_values_batch,target_q_values_batch)
        # Do backward pass
        self.loss.backward()
        self.optimizer.step()
        # save loss
        self.LOSS += [self.loss.item()]
        # reset memory
        self.MEMORY_ = self.MEMORY
        self.MEMORY = [[],[],[],[],[]]
    
    ## reset object
    def RESET(self, PROBA):
        GRAPH = self.NET.NEXT_GEN(-1)
        XY_TUPLE = (self.X,self.Y)
        if np.random.choice((False,True), 1, p=[PROBA,1-PROBA])[0]:
            return Q_AGENT(*self.ARG, NET = GRAPH, COOR = XY_TUPLE)
        else :
            return Q_AGENT(*self.ARG, MODEL = self.MODEL, NET = GRAPH, COOR = XY_TUPLE)
    
    ## mutation
    def MUTATION(self, MUT = None):
        # mutate graph
        GRAPH = self.NET.NEXT_GEN(MUT)
        return Q_AGENT(*self.ARG, NET = GRAPH)
    
    ## control group
    def CONTROL_NETWORK(self, IO) :
        """
        For Lyfe problem : not generalized
        """
        # init number of connection per layer
        """NB_H_LAYER = 2"""
        """NB_C_P_LAYER = int(np.sqrt(self.IO[0]) + np.sqrt(self.IO[1]))"""
        # network equivalence --> passer à 17 ?
        NET = np.array([[-1, 3, 4, 32, [[2,0],[2,1],[2,2],[2,3]]],
                        [ 1, 4, IO[0], 10, [[0,i] for i in range(IO[0])]],
                        [ 2, 4, 4, 20, [[1,0],[1,1],[1,2],[1,3]]]])
        # Listing
        LIST_C = np.array([[0,0,i] for i in range(IO[0])]+
                          [[10,1,0],[10,1,1],[10,1,2],[10,1,3],
                          [20,2,0],[20,2,1],[20,2,2],[20,2,3]])
        return [IO, NET.copy(), LIST_C.copy()]

if __name__ == '__main__' :
    # test combinaison strategy
    from scipy.special import comb
    print(comb(25, 17, exact=False))
    # init
    ARG = ((9,3),25, 16, 16, 12, 2)
    q = Q_AGENT(*ARG, CTRL=True)
    #print(q.NEURON_LIST)
    #Mutate
    DXY = (np.ones((5,5))/25,np.ones((3,3))/9)
    for i in range(10) :
        print(i)
        p = q.MUTATION(DXY)
        #print(p.NEURON_LIST)
    # add input
    IN = np.zeros((5,5))
    IN[tuple(map(tuple, (p.X+[2,2]).T))] = 1
    plt.imshow(IN)
    # memory
    MEMORY = [[], [], [], [], []]
    MEMORY[0] = torch.tensor(np.random.random((16,9)), dtype=torch.float) # old_state
    MEMORY[1] = torch.tensor(np.random.randint(0,3, (16,1)),  dtype=torch.long) #.unsqueeze(1) # action
    MEMORY[2] = torch.tensor(np.random.random((16,9)), dtype=torch.float) # new_state
    MEMORY[3] = torch.tensor(np.random.randint(-10,10, (16))) # reward
    MEMORY[4] = torch.tensor(np.random.randint(0,2, (16)), dtype=torch.int) # DONE
    q.MEMORY = MEMORY
    # optim test
    q.OPTIM()