#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 23:05:53 2021
@author: fabien
"""

import numpy as np
import pandas as pd, os

import EXTRA_FUNCTION as EF
from scipy.ndimage import filters

################################ SAVE EXP INFO & pre-treatment
class LOG_INFO():
    def __init__(self, PL_LIST, ENV_LIST, GEN, CYCL, RULE, BATCH, NTIME):
        # parameter & score
        self.RULE = RULE
        self.BATCH = BATCH
        self.N_TIME = NTIME
        self.NB_CYCLE = CYCL
        self.NB_SEED = int(np.sqrt(len(PL_LIST))-1)
        self.NB_CONTROL = int(np.rint(np.power(len(PL_LIST), 1./4))) # see main ... not generalized
        self.SCORE = []
        self.LOSS = []
        # exp info
        self.DF_1 = pd.DataFrame(columns=['ID','GEN','TYPE','TREE','MAP_SIZE','NEURON_LIST'])
        self.DF_2 = pd.DataFrame(columns=['ID','AGENT_POS','PNJ_POS','TAG_STATE','SCORE', 'LOSS', 'RANK'])
        self.DF = self.DF_1 + self.DF_2
        # Init dataframe
        self.START_CYCLE(PL_LIST, ENV_LIST, GEN)
    
    def SAVE_CSV(self, TIME) :
        FOLDER = 'OUT'
        if(not os.path.isdir(FOLDER)): os.makedirs(FOLDER)
        if type(TIME) != type('') :
            TIME = '_'
        self.DF.to_csv(FOLDER + os.path.sep + 'LYFE_RULE_' + str(self.RULE) +"_EXP_"+ TIME + '_.csv', sep=';', index=False)
    
    def START_CYCLE(self, PLAYS_LIST, ENV_LIST, GEN, SLC_LIST = None):
        # Listing
        ID_ = []
        GEN_ = []
        TYPE = []
        TREE = []
        MAP_SIZE = []
        NEURON_LIST = []
        # survival, legacy, challenger
        if SLC_LIST != None :
            c, s, l, f = -1, 0, 1, self.DF_1['TREE'].max()[0]+1
        # Loop
        for i in range(len(PLAYS_LIST)) :
            ID_ += [self.DF_1.shape[0]+i]
            GEN_ += [GEN]
            if SLC_LIST == None :
                TREE += [[i]]
                if i < self.NB_CONTROL :
                    TYPE += [-1]
                else : 
                    TYPE += [0]
            else :
                L = SLC_LIST[i]
                L_ = 0
                # Survival
                if L[0] == 's' :
                    TREE += [[L[1]]+[s]]
                    TYPE += [1]
                # Legacy
                elif L[0] == 'l' :
                    TYPE += [2]
                    if L[1] == L_ :
                        TREE += [[L[1]]+[l]]
                        l += 1
                    else :
                        L_ = L[1]
                        l = 1
                        TREE += [[L[1]]+[l]]
                # Challenger
                elif L[0] == 'f' :
                    TYPE += [3]
                    TREE += [[f]]
                    f += 1
                # control
                elif L[0] == 'c' :
                    TYPE += [-1]
                    TREE += [[c]]
            MAP_SIZE += [ENV_LIST[i].MAP_SIZE]
            NEURON_LIST += [PLAYS_LIST[i].NEURON_LIST.tolist()]
        # Array construction
        ARRAY = np.zeros((len(PLAYS_LIST),self.DF_1.columns.size), dtype=object)
        for i in range(len(PLAYS_LIST)):
            ARRAY[i,0] = ID_[i]
            ARRAY[i,1] = GEN_[i]
            ARRAY[i,2] = TYPE[i]
            ARRAY[i,3] = TREE[i]
            ARRAY[i,4] = MAP_SIZE[i]
            ARRAY[i,5] = NEURON_LIST[i]
        # UPDATE DF1
        DF_1_NEW = pd.DataFrame(ARRAY, columns=list(self.DF_1))
        self.DF_1 = self.DF_1.append(DF_1_NEW, ignore_index=True)
    
    def FINISH_CYCLE(self, ENV_LIST, SCORE, RANK, AG_LOSS, SHOWSCORE=True):
        self.SCORE += [np.array(SCORE)[None]]
        self.LOSS += [np.concatenate([np.array(a.LOSS)[None] for a in AG_LOSS])]
        # fast score plot
        if SHOWSCORE and len(self.SCORE) > 1 :
            self.SCORE_PLOT()
            self.LOSS_PLOT()
        # Listing
        ID = np.arange(self.DF_2.shape[0], self.DF_2.shape[0] + len(ENV_LIST))
        AGENT_POS = []
        PNJ_POS = []
        TAG_STATE = []
        # Loop 
        for e in ENV_LIST :
            AGENT_POS += [e.AG_LIST]
            PNJ_POS += [e.PNJ_LIST]
            TAG_STATE += [e.IT_LIST]
        # Array construction
        ARRAY = np.zeros((len(ENV_LIST),self.DF_2.columns.size), dtype=object)
        for i in range(len(ENV_LIST)):
            ARRAY[i,0] = ID[i]
            ARRAY[i,1] = AGENT_POS[i]
            ARRAY[i,2] = PNJ_POS[i]
            ARRAY[i,3] = TAG_STATE[i]
            ARRAY[i,4] = SCORE[i]
            ARRAY[i,5] = AG_LOSS[i].LOSS
            ARRAY[i,6] = RANK[i]
        # UPDATE DF2
        DF_2_NEW = pd.DataFrame(ARRAY, columns=list(self.DF_2))
        self.DF_2 = pd.concat([self.DF_2, DF_2_NEW])
        # MERGE DF1 + DF2 (pointer)
        self.DF = pd.merge(self.DF_1, self.DF_2, on="ID")
    
    def LOSS_PLOT(self):
        CURVE = np.concatenate(self.LOSS, axis=1)
        self.PLOT(CURVE, self.NB_CYCLE*self.N_TIME, "LOSS","BATCH", SIGMA=3)
        self.POPPLOT(CURVE, self.NB_CYCLE*self.N_TIME)
        
    def SCORE_PLOT(self):
        CURVE = np.concatenate(self.SCORE).T
        self.PLOT(CURVE, self.NB_CYCLE, "SCORE","GEN")
        self.POPPLOT(CURVE, self.NB_CYCLE)
    
    def POPPLOT(self, CURVE, XMAX) :
        VALUE = np.zeros((CURVE.shape[0], XMAX))
        VALUE[:] = np.mean(CURVE)
        VALUE[:,:CURVE.shape[1]] = CURVE
        EF.FAST_IMSHOW([VALUE,VALUE[:self.NB_CONTROL+self.NB_SEED]])
    
    def PLOT(self, CURVE, XMAX, Y, X, SIGMA=1):
        curve_list = [  filters.gaussian_filter1d(CURVE[:self.NB_CONTROL].mean(0),SIGMA), 
                        filters.gaussian_filter1d(CURVE[self.NB_CONTROL:self.NB_CONTROL+self.NB_SEED].mean(0),SIGMA), 
                        filters.gaussian_filter1d(CURVE[self.NB_CONTROL:-self.NB_SEED].mean(0),SIGMA), 
                        filters.gaussian_filter1d(CURVE[-self.NB_SEED:].mean(0),SIGMA)]
        std_list = [    filters.gaussian_filter1d(CURVE[:self.NB_CONTROL].std(0),SIGMA),
                        filters.gaussian_filter1d(CURVE[self.NB_CONTROL:self.NB_CONTROL+self.NB_SEED].std(0),SIGMA), 
                        filters.gaussian_filter1d(CURVE[:self.NB_CONTROL:-self.NB_SEED].std(0),SIGMA), 
                        filters.gaussian_filter1d(CURVE[-self.NB_SEED:].std(0),SIGMA)]
        EF.FAST_PLOT(curve_list,std_list,['CTRL','BEST','EVOLUTION','RANDOM'], 
                     'LYFE', Y,X, self.RULE, self.BATCH, self.N_TIME, self.NB_SEED, XMAX)