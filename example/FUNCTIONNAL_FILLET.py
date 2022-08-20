#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:09:28 2021
@author: fabien
"""

# ML module
import numpy as np, pylab as plt
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

import pickle, datetime, os

from tqdm import tqdm

# networks construction
from GRAPH_EAT import GRAPH_EAT
from pRNN_GEN import pRNN

# extra function
import EXTRA_FUNCTION as EF

# calculation (multi-cpu) and data (namedtuple) optimisation
#import multiprocessing, collections

# control net
class CTRL_NET(nn.Module):
    def __init__(self, IO):
        super(CTRL_NET, self).__init__()
        I,O = IO
        H = int(np.sqrt(I))
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

# ff module
class model():
    def __init__(self, IO, SAMPLE_SIZE, BATCH_SIZE, EPOCH, NB_GEN, NB_SEEDER, RNN = False, DATA_XPLT = 0.5, LEARNING_RATE = 1e-6, MOMENTUM = 0.5):
        # Parameter
        self.IO = IO
        self.N = SAMPLE_SIZE
        self.BATCH = BATCH_SIZE
        self.EPOCH = EPOCH
        self.NB_GEN = NB_GEN
        self.NB_SEEDER = NB_SEEDER**2
        self.LR = LEARNING_RATE
        self.MM = MOMENTUM
        # generate first ENN step
        self.GRAPH_LIST = [GRAPH_EAT([self.IO, 1], None) for n in range(self.NB_SEEDER-1)]
        self.SEEDER_LIST = [CTRL_NET(self.IO)]
        for g in self.GRAPH_LIST :
            NEURON_LIST = g.NEURON_LIST
            self.SEEDER_LIST += [pRNN(NEURON_LIST, self.BATCH, self.IO[0], STACK=RNN)]
        self.PARENT = [(-1*np.ones(self.NB_SEEDER))[None]]
        # best seeder model
        self.BEST_MODEL = 0
        self.OPTIM_BEST = 0
        self.BEST_CRIT = nn.CrossEntropyLoss()
        self.LOSS_BEST = 0
        self.BEST_WEIGHT = []
        # generate loss-optimizer
        self.OPTIM = [torch.optim.SGD(s.parameters(), lr=LEARNING_RATE,momentum=MOMENTUM) for s in self.SEEDER_LIST]
        self.CRITERION = [nn.CrossEntropyLoss() for n in range(self.NB_SEEDER)]
        self.LOSS = self.NB_SEEDER*[0]
        # calculate nb batch per generation
        self.NB_BATCH_P_GEN = int((DATA_XPLT*self.N*self.EPOCH)/(self.NB_GEN*self.BATCH))
        # selection and accuracy
        self.SCORE_LOSS = []
        self.ACCUR = [] # in %
        self.BEST_SCORE_LOSS = []
        # for next gen (n-plicat) and control group
        self.NB_CONTROL = 1 # always (preference)
        self.NB_CHALLENGE = int(np.sqrt(self.NB_SEEDER)-self.NB_CONTROL)
        self.NB_SURVIVOR = self.NB_CHALLENGE # square completion
        self.NB_CHILD = int(np.sqrt(self.NB_SEEDER)-1) # FITNESS
    
    def fit(self, DATA, LABEL):
        # gen loop
        for o in tqdm(range(self.NB_GEN)):
            DATA,LABEL = shuffle(DATA,LABEL)
            P = (self.NB_GEN - o)/(2*self.NB_GEN) # proba
            # compilation
            for n in range(self.NB_BATCH_P_GEN):
                data = torch.tensor(DATA[n*self.BATCH:(n+1)*self.BATCH].reshape(-1,self.IO[0]), dtype=torch.float)
                target = torch.tensor(LABEL[n*self.BATCH:(n+1)*self.BATCH]).type(torch.LongTensor)
                # seed
                for s in range(self.NB_SEEDER):
                    self.OPTIM[s].zero_grad()
                    output = self.SEEDER_LIST[s](data)
                    self.LOSS[s] = self.CRITERION[s](output,target)
                    self.LOSS[s].backward()
                    self.OPTIM[s].step()
                # score loss
                self.SCORE_LOSS += [torch.tensor(self.LOSS).numpy()[None]]
            # score accuracy
            train_idx = np.random.randint(self.N, size=self.BATCH)
            dt_train = torch.tensor(DATA[train_idx].reshape((-1,self.IO[0])), dtype=torch.float)
            tg_train = torch.tensor(LABEL[train_idx])
            max_idx = self.predict(dt_train, False) 
            self.ACCUR += [((max_idx == tg_train).sum(1)/self.BATCH).numpy()[None]]
            # evolution
            SCORE_LIST = ((1-self.ACCUR[-1]).squeeze())*(self.SCORE_LOSS[-1].squeeze()) # square effect
            ## fitness (in accuracy test)
            ORDER = np.argsort(SCORE_LIST[self.NB_CONTROL:]).astype(int)
            # control
            CTRL = self.SEEDER_LIST[:self.NB_CONTROL]
            PARENT = [0]*self.NB_CONTROL
            # survivor (reset weight or not)
            BEST = []
            B_G_ = []
            B_I = []
            for i in ORDER[:self.NB_SURVIVOR] :
                B_G_ += [self.GRAPH_LIST[i]]
                if np.random.choice((True,False), 1, p=[P,1-P]):
                    BEST += [self.SEEDER_LIST[self.NB_CONTROL:][i]]
                else :
                    BEST += [pRNN(B_G_[-1].NEURON_LIST, self.BATCH, self.IO[0])]
                PARENT += [i+1]
                B_I += [i+1]
            # mutation
            MUTS = []
            M_G_ = []
            for g,j in zip(B_G_,B_I) :
                for i in range(self.NB_CHILD):
                    M_G_ += [g.NEXT_GEN()]
                    MUTS += [pRNN(M_G_[-1].NEURON_LIST, self.BATCH, self.IO[0])]
                    PARENT += [j]
            # challenger
            NEWS = []
            N_G_ = []
            for n in range(self.NB_CHALLENGE) :
                N_G_ += [GRAPH_EAT([self.IO, 1], None)]
                NEWS += [pRNN(N_G_[-1].NEURON_LIST, self.BATCH, self.IO[0])]
                PARENT += [-1]
            # update
            self.SEEDER_LIST = CTRL + BEST + MUTS + NEWS
            self.GRAPH_LIST = B_G_ + M_G_ + N_G_
            self.PARENT += [np.array(PARENT)[None]]
            # generate loss-optimizer
            self.OPTIM = [torch.optim.SGD(s.parameters(), lr=self.LR,momentum=self.MM) for s in self.SEEDER_LIST]
            self.CRITERION = [nn.CrossEntropyLoss() for n in range(self.NB_SEEDER)]
        # compact evolution data
        self.SCORE_LOSS = np.concatenate(self.SCORE_LOSS).T
        self.ACCUR = np.concatenate(self.ACCUR).T
        self.PARENT = np.concatenate(self.PARENT).T
        # best loop weight optimization
        self.BEST_MODEL = pRNN(self.GRAPH_LIST[ORDER[0]].NEURON_LIST, self.BATCH, self.IO[0])
        self.OPTIM_BEST = torch.optim.SGD(self.BEST_MODEL.parameters(), lr=self.LR,momentum=self.MM)
        for i in tqdm(range(self.NB_GEN)):
            DATA,LABEL = shuffle(DATA,LABEL)
            for n in range(self.NB_BATCH_P_GEN):
                data = torch.tensor(DATA[n*self.BATCH:(n+1)*self.BATCH].reshape(-1,self.IO[0]), dtype=torch.float)
                target = torch.tensor(LABEL[n*self.BATCH:(n+1)*self.BATCH]).type(torch.LongTensor)
                self.OPTIM_BEST.zero_grad()
                output = self.BEST_MODEL(data)
                self.LOSS_BEST = self.BEST_CRIT(output,target)
                self.LOSS_BEST.backward()
                self.OPTIM_BEST.step()
                # score loss
                self.BEST_SCORE_LOSS += [self.LOSS_BEST.detach().numpy()[None]]
        self.BEST_SCORE_LOSS = np.concatenate(self.BEST_SCORE_LOSS)
        # Extract learned weight
        self.BEST_WEIGHT = list(self.BEST_MODEL.parameters())
        # save object
        if(not os.path.isdir('OUT')): os.makedirs('OUT')
        TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filehandler = open("OUT"+os.path.sep+"MODEL_"+TIME+".obj", 'wb')
        pickle.dump(self, filehandler); filehandler.close()
        
    def predict(self, DATA, WITH_VAL = True):
        with torch.no_grad():
            log_prob = []
            for s in self.SEEDER_LIST :
                log_prob += [s(DATA)[None]]
        proba = torch.exp(torch.cat(log_prob))
        max_values, max_index = torch.max(proba,2)
        if WITH_VAL :
            return max_values, max_index
        else : 
            return max_index

### ALGO MNIST PART
    
if __name__ == '__main__' :
    LOAD = True
    Filename = 'MODEL_20211202_164612.obj' #49
    
    #Filename = 'MODEL_20211202_164612.obj' #49
    #Filename = 'MODEL_20211203_174344.obj' #25
    #Filename = 'MODEL_20211214_212341.obj' #49 + 10g + 10 epoch ## REDUCE TIME BATCH !!! (1000 max)
    # module for test
    from tensorflow.keras.datasets import mnist
    # import mnist
    (X_train_data,Y_train_data),(X_test_data,Y_test_data) = mnist.load_data()
    # shuffle data
    X_train_data,Y_train_data = shuffle(X_train_data,Y_train_data)
    plt.imshow(X_train_data[0]); print(Y_train_data[0]); plt.show(); plt.close()
    # data resizing
    X_train_data, X_test_data = X_train_data[:,::2,::2], X_test_data[:,::2,::2]
    plt.imshow(X_train_data[0]); plt.show(); plt.close()
    # data info
    N, x, y = X_train_data.shape ; I = x*y
    O = np.unique(Y_train_data).size
    # parameter
    PRCT = 1. # data percet using
    BATCH, EPOCH = 25, 10
    NB_GEN, NB_SEED = 3, 10
    # init&train or load
    if LOAD :
        with open('OUT'+os.path.sep+Filename, 'rb') as f:
            MODEL = pickle.load(f)
    else :
        MODEL = model((I,O), N, BATCH, EPOCH, NB_GEN,NB_SEED, DATA_XPLT = PRCT)
        # training
        MODEL.fit(X_train_data,Y_train_data)
    # seeder position
    TYPE = ['CTRL','PARENT','CHILD','RANDOM']
    LENGHT = np.array([MODEL.NB_CONTROL, MODEL.NB_SURVIVOR+MODEL.NB_CHILD**2, MODEL.NB_CHALLENGE])
    LENGHT = np.array([MODEL.NB_CONTROL, MODEL.NB_SURVIVOR, MODEL.NB_CHILD**2, MODEL.NB_CHALLENGE])
    START = np.array([0]+np.cumsum(LENGHT)[:-1].tolist())
    ## preprocess {a, b, d = 1050, 50, 0.225}
    a, b, d = 1050, 50, 0.5
    ## result
    EF.FAST_IMSHOW([MODEL.SCORE_LOSS], ['LOSS'])
    curve_list, std_list = EF.FAST_CURVE_CONSTRUCT(MODEL.SCORE_LOSS, MODEL.BEST_SCORE_LOSS[None], [START,LENGHT], 2) # 5 for big, 2 otherwise
    # lenght of curve reduction (to big in svg)
    x_pos = np.linspace(0,len(curve_list[0])-1,1000).astype('int')
    curve_, std_ = [], []
    for c,s in zip(curve_list, std_list) :
        curve_ += [c[x_pos]]
        std_ += [s[x_pos]]
    # filter
    x_factor = 0.5-((x_pos-a) / (b + np.abs(x_pos-a)))/4 + d
    plt.plot(x_factor)
    std_[1] = std_[1]*x_factor
    # plot
    EF.FAST_PLOT(curve_,std_, TYPE +['BEST'], 'MNIST', 'LOSS','BATCH', STD = 2, yaxis=[0.,30], x_reduce=x_pos)
    # predict
    X_test_data,Y_test_data = shuffle(X_test_data,Y_test_data)
    X_torch = torch.tensor(X_test_data[:10].reshape((-1,x*y)), dtype=torch.float)
    max_v, max_i = MODEL.predict(X_torch)
    Y_torch = torch.tensor(Y_test_data[:10])
    print("Predicted Score =", (max_i == Y_torch).sum(1))
    # evolution graph
    PARENT = MODEL.PARENT.T
    s = PARENT.shape
    node = np.arange(np.prod(s))
    # network
    pos, G = EF.IMLINEAGE_2_GRAPH(node,PARENT)
    G, edges_size, node_size, SHORT_PATH = EF.ADD_PATH(node,G)
    ## show edge
    E_ = np.log(edges_size.reshape(s).T+1)
    E_ = (E_-E_.min())/(E_.max()-E_.min())
    curve_e, std_e = EF.FAST_CURVE_CONSTRUCT(E_, [E_.mean(0), E_.std(0)], [START,LENGHT], 0.5, BONUS=True)
    EF.FAST_PLOT(curve_e, std_e,['CTRL','EVOLUTION','RANDOM','MEAN'], 'MNIST', 'EDGES','GEN', yaxis=[0.,1.])
    # show node
    N_ = np.sqrt(node_size.reshape(s).T)
    #N_ = (N_-N_.min())/(N_.max()-N_.min())
    #curve_n, std_n = EF.FAST_CURVE_CONSTRUCT(N_, [N_.mean(0), N_.std(0)/(np.pi)], [START,LENGHT], 0.5)
    curve_n, std_n = EF.FAST_CURVE_CONSTRUCT(N_,'', [START,LENGHT], 0.5)
    YMAX = np.max(curve_n)+0.5 #np.sqrt(N_.shape[1])
    #EF.FAST_PLOT(curve_n, std_n, TYPE+['MEAN'], 'MNIST', 'NODES','GEN', yaxis=[1.,YMAX])
    EF.FAST_PLOT(curve_n, std_n, TYPE, 'MNIST', 'NODES','GEN', yaxis=[1.,YMAX], STEP=True)
    # distribution
    EF.FAST_IMSHOW([E_,N_], ['EDGES','NODE'])
    #draw evolution
    """
    H = G.to_undirected()
    nodes = nx.draw_networkx_nodes(H, posD, node_size=0.1, alpha=0.5)
    edges = nx.draw_networkx_edges(H, posD, edge_color=edges_size, alpha=0.5)
    plt.savefig("OUT/Tree.svg")
    plt.show(); plt.close()
    """
    # dendrogram
    """
    from scipy.cluster import hierarchy
    mx = nx.to_pandas_adjacency(G).values
    Z = hierarchy.linkage(mx)
    dn = hierarchy.dendrogram(Z)
    plt.show()
    """
