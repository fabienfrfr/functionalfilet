#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:19:36 2021
@author: fabien
"""

import numpy as np

from functionalfilet.graph_gen import GRAPH

import copy

################################ GRAPH Evolution Augmenting Topology
class GRAPH_EAT(GRAPH):
    def __init__(self, IO, P_MIN=1, NET=None):
        if NET == None :
            # first generation
            # Inheritance of Graph_gen
            super().__init__(IO, P_MIN)
        else : 
            self.IO, self.NEURON_LIST, self.LIST_C = NET
        
    def NEXT_GEN(self, MUT = None):
        # copy of module (heritage)
        IO, NEURON_LIST, LIST_C = self.IO, copy.deepcopy(self.NEURON_LIST), copy.deepcopy(self.LIST_C)
        # adding mutation (variation)
        if MUT == None :
            MUT = np.random.choice(7)
        #print('Mutation type : '+str(MUT))
        if MUT == 0 :
            # add connection
            NEURON_LIST = self.ADD_CONNECTION(NEURON_LIST, LIST_C)
        elif MUT == 1 :
            # add neuron
            NEURON_LIST, LIST_C = self.ADD_NEURON(NEURON_LIST)
        elif MUT == 2 :
            # add layers
            NEURON_LIST, LIST_C = self.ADD_LAYERS(NEURON_LIST, LIST_C)
        elif MUT == 3 :
            # cut doublon connect neuron
            NEURON_LIST = self.CUT_CONNECTION(NEURON_LIST)
        elif MUT == 4 :
            # cut neuron
            NEURON_LIST, LIST_C = self.CUT_NEURON(NEURON_LIST, LIST_C)
        elif MUT == 5 :
            # transpose 1 connection
            NEURON_LIST = self.C_TRANSPOSITION(NEURON_LIST)
        elif MUT == 6 :
            # permute 2 connection
            NEURON_LIST = self.C_PERMUTATION(NEURON_LIST)
        # return neuronList with mutation or not
        return GRAPH_EAT(None,NET=[IO, NEURON_LIST.copy(), LIST_C.copy()])
    
    ### MUTATION PART
    def ADD_CONNECTION(self, NEURON_LIST, LIST_C) :
        # add Nb connect
        idx = np.random.randint(NEURON_LIST.shape[0])
        NEURON_LIST[idx,2] += 1
        # add element list
        idx_ = np.random.randint(LIST_C.shape[0])
        NEURON_LIST[idx,-1] += [LIST_C[idx_, 1:].tolist()]
        return  NEURON_LIST

    def ADD_NEURON(self, NEURON_LIST):
        NB_LAYER = NEURON_LIST[:,0].max()
        # add neuron in one layers
        if NB_LAYER == 1 : IDX_N = 1
        else : IDX_N = np.random.randint(1,NB_LAYER)
        NEURON_LIST[IDX_N,1] += 1
        idx_new, idx_c_new = NEURON_LIST[IDX_N,0], NEURON_LIST[IDX_N,1]-1
        # add connection
        IDX_C = np.random.randint(NB_LAYER)
        NEURON_LIST[IDX_C, 2] += 1
        NEURON_LIST[IDX_C,-1] += [[idx_new, idx_c_new]]
        # update list_connection
        LIST_C = self.LISTING_CONNECTION(NEURON_LIST.shape[0]-1, NEURON_LIST[:,:-1])
        return NEURON_LIST, LIST_C
    
    def C_PERMUTATION(self, NEURON_LIST):
        ## only 2 neuron permut
        choosable = np.where(NEURON_LIST[:,2]>1)[0]
        if len(choosable) <= 1 :
            return NEURON_LIST
        # select 2 layers
        i,j = np.random.choice(choosable, 2, replace=False)
        # select index (first not included)
        p = np.random.randint(NEURON_LIST[i,2]-1)+1
        q = np.random.randint(NEURON_LIST[j,2]-1)+1
        # save values
        c = NEURON_LIST[i,-1][p]
        d = NEURON_LIST[j,-1][q]
        # permute
        NEURON_LIST[i,-1][p] = d
        NEURON_LIST[j,-1][q] = c
        return NEURON_LIST
    
    def C_TRANSPOSITION(self, NEURON_LIST):
        ## only 2 neuron permut
        choosable = np.where(NEURON_LIST[:,2]>1)[0]
        if len(choosable) <= 1 :
            return NEURON_LIST
        # select 1 layers and new loc
        i = np.random.choice(choosable, 1)[0]
        possible_trnp = np.setdiff1d(np.arange(NEURON_LIST.shape[0]), i)        
        j = np.random.choice(possible_trnp, 1)[0]
        # select index (first not included)
        p = np.random.randint(NEURON_LIST[i,2]-1)+1
        # save values
        c = NEURON_LIST[i,-1][p]
        # transpose and delete
        NEURON_LIST[j,-1] += [c]
        del(NEURON_LIST[i,-1][p])
        # update nb param
        NEURON_LIST[i,2] -= 1
        NEURON_LIST[j,2] += 1
        return NEURON_LIST
    
    def ADD_LAYERS(self, NEURON_LIST, LIST_C):
        # new one neuron layers test
        idx_new = NEURON_LIST[:,0].max() + 1
        max_pos = NEURON_LIST[:,3].max()
        possible_pos = np.setdiff1d(np.arange(1,max_pos), NEURON_LIST[:,3])
        if len(possible_pos) == 0:
            return NEURON_LIST, LIST_C
        # add position and init
        POS_X_new = np.random.choice(possible_pos, 1, replace=False)[0]
        NEW_NEURON = np.array([idx_new, 1, 1, POS_X_new, []])
        # connection of new neuron input (not downstream neuron necessary)
        IDX_C = np.where(LIST_C[:,0] < max_pos)[0]
        idx_c = IDX_C[np.random.randint(IDX_C.shape[0])]
        list_c = LIST_C[idx_c, 1:] ; #print(list_c)
        NEW_NEURON[-1] = [list(list_c)]
        # adding connection (not downstream neuron necessary : only in first constructions)
        IDX_N = np.where(NEURON_LIST[:,3] < max_pos)[0]
        idx_n = IDX_N[np.random.randint(IDX_N.shape[0])]
        NEURON_LIST[idx_n, 2] += 1
        NEURON_LIST[idx_n, -1] += [[idx_new, 0]]
        # add layers and update list
        NEURON_LIST = np.concatenate((NEURON_LIST,NEW_NEURON[None]), axis=0)
        LIST_C = self.LISTING_CONNECTION(NEURON_LIST.shape[0]-1, NEURON_LIST[:,:-1])
        return NEURON_LIST, LIST_C
    
    def CUT_CONNECTION(self, NEURON_LIST):
        # listing of connection
        CONNECT_DATA = self.CONNECTED_DATA(NEURON_LIST)
        # choose connect duplicate (doublon : ! min connect, otherwise : return)
        c_u, ret = np.unique(CONNECT_DATA[:,2:], axis=0, return_counts=True)
        idx_doublon = np.where(ret > 1)[0]
        if idx_doublon.shape == (0,) :
            return NEURON_LIST
        c_2_cut = c_u[idx_doublon[np.random.randint(len(idx_doublon))]]
        # find cut connection (! 1st link : vestigial, otherwise : return)
        IDX_CD = np.where((CONNECT_DATA[:,1] !=0)*(CONNECT_DATA[:,2:] == c_2_cut).all(axis=1))[0]
        if IDX_CD.shape == (0,) :
            return NEURON_LIST
        idx_cd = IDX_CD[np.random.randint(IDX_CD.shape)][0]
        idx, idx_ = CONNECT_DATA[idx_cd, :2]
        # update neuronlist
        IDX = np.where(NEURON_LIST[:,0] == idx)[0]
        NEURON_LIST[IDX,2] -= 1
        del(NEURON_LIST[IDX,-1][0][int(idx_)])
        return NEURON_LIST
    
    def CUT_NEURON(self, NEURON_LIST, LIST_C):
        # listing of connection
        CONNECT_DATA = self.CONNECTED_DATA(NEURON_LIST)
        ## find possible neuron (no ones connection)
        c_n, idx_first, ret = np.unique(CONNECT_DATA[:,0], return_counts=True, return_index=True)
        idx_ones = c_n[np.where(ret == 1)[0]]
        # ones verif (unselect neuron with one connection)
        if idx_ones.shape[0] != 0 :
            # select index
            bool_o  = np.any([CONNECT_DATA[:,0] == i for i in idx_ones], axis = 0)
            Co = CONNECT_DATA[bool_o,2:]
            # select doublon
            bool_o_ = np.any([(CONNECT_DATA[:,2:] == d).all(axis=1) for d in Co], axis=0)
            # choose neuron
            C_ = CONNECT_DATA[np.invert(bool_o_), 2:]
        else :
            # choose neuron
            C_ = CONNECT_DATA[:, 2:]
        # delete first link (vestigial)
        Cf = CONNECT_DATA[idx_first, 2:]
        bool_f_ = np.any([(C_ == f).all(axis=1) for f in Cf], axis=0)
        bool_f = np.invert(bool_f_)
        C_ = C_[bool_f]
        # delete input
        C_ = C_[C_[:,0] != 0]
        # return if no connection
        if C_.shape[0] == 0 :
            return NEURON_LIST, LIST_C
        # choise index
        idx, idx_ = C_[np.random.randint(C_.shape[0])]
        IDX = np.where(NEURON_LIST[:,0] == idx)[0]
        # return if no neuron
        NEW_NB_NEURON = NEURON_LIST[IDX, 1] - 1
        if NEW_NB_NEURON == 0 :
            return NEURON_LIST, LIST_C
        # remove one neuron number
        NEURON_LIST[IDX, 1] = NEW_NB_NEURON
        #print(idx, idx_)
        # update list of neuron
        for n in NEURON_LIST :
            list_c = np.array(n[-1])
            # boolean element
            egal = (list_c == [idx, idx_]).all(axis=1)
            sup_ = (list_c[:,0] == idx) * (list_c[:,1] >= idx_)
            # change connection (shift empty)
            if sup_.any() :
                list_c[sup_,1] -= 1
                list_c = list_c[np.invert(egal)]
            # update neuronlist
            n[-1] = list_c.tolist()
            n[2] = len(n[-1])
        # update connection list
        LIST_C = self.LISTING_CONNECTION(NEURON_LIST.shape[0]-1, NEURON_LIST[:,:-1])
        return NEURON_LIST, LIST_C
    
    def CONNECTED_DATA(self, NEURON_LIST):
        CONNECT_DATA = []
        for n in NEURON_LIST :
            idx = n[0]*np.ones((n[2],1))
            idx_ = np.arange(n[2])[:,None]
            c = np.array(n[-1])
            CONNECT_DATA += [np.concatenate((idx,idx_,c),axis=1)]
        CONNECT_DATA = np.concatenate(CONNECT_DATA)
        return CONNECT_DATA


if __name__ == '__main__' :
    g = GRAPH_EAT([9, 3, 1], None)
    nl = g.NEURON_LIST
    print(nl)
    print('init : ' + str([len(nl), sum(nl[:,1]), sum(nl[:,2])]))
    mut = ['addco', 'addn', 'addl', 'delco', 'deln','transpc','permc']
    for i in range(7):
        h = g.NEXT_GEN(i)
        nl = h.NEURON_LIST
        print(mut[i] + ' : ' + str([len(nl), sum(nl[:,1]), sum(nl[:,2])]))
        #print(nl)