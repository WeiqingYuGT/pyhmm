"""
  -------------------------------- (C) ---------------------------------
myhmm_scaled.py
Author: Anantharaman Narayana Iyer
Date: 1 March 2015

                         Author: Anantharaman Palacode Narayana Iyer
                         <narayana.anantharaman@gmail.com>

  Distributed under the BSD license:

    Copyright 2010 (c) Anantharaman Palacode Narayana Iyer, <narayana.anantharaman@gmail.com>

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

        * Redistributions of source code must retain the above
          copyright notice, this list of conditions and the following
          disclaimer.

        * Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials
          provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
    OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
    THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.

    This module implements the HMM algorithms as described in Rabiner's book.
    In order to avoid underflows for long observation sequences scaling is implemented in forward and backward functions
    Also support for sequences with multiple observation vectors (as is needed for speech) is implemented

    NOTE: Some errata is reported from the original paper of Rabiber
    See: http://alumni.media.mit.edu/~rahimi/rabiner/rabiner-errata/rabiner-errata.html
    The code in this module applies the corrections mentioned in this errata
    
"""
#import json
#import os
import sys
import math
import numpy as np
import heapq

class MyHmmContScaled(object):
    # base class for different HMM models - implements Rabiner's algorithm for scaling to avoid underflow
    def __init__(self, model):
        # model is (A, B, pi) where A = Transition probs, B = Emission Probs, pi = initial distribution
        # a model can be initialized to random parameters using a json file that has a random params model
        def eachInv(x):
            return np.matrix(x).I
        
        if model == None:
            print "Fatal Error: You should provide the initial model"
            sys.exit()
#        self.model = json.loads(open(model_name).read())["hmm"]
        self.model = model
        self.A = np.array(self.model["A"])
        self.states = range(self.A.shape[0]) # get the list of states
        self.N = len(self.states) # number of states of the model
        self.means = np.array(self.model["means"])
        self.covars = np.array(self.model["cov"])
        self.covarsInv = np.array([eachInv(x) for x in self.covars])
        self.M = self.means.shape[0] # number of states of the model
        self.d = self.means.shape[1]
        self.pi = self.model["pi"]
        self.min_std = self.model['mstd']
        
        
        return

    def _pdf(self, x, mean, covarInv):
        inv_covar_det = np.linalg.det(covarInv)
        c = ( ( (2.0*np.pi)**(-float(self.d)/2) * (inv_covar_det)**(0.5)))
        pdfval = c * np.exp(-0.5 * np.dot( np.dot((x-mean),covarInv), (x-mean)))
        return max(pdfval,1e-20)
    
    def _lpdf(self, x, mean, covarInv):
        inv_covar_det = np.linalg.det(covarInv)
        lpdfval = -float(self.d)/2*math.log(2*np.pi)+0.5*math.log(inv_covar_det)-\
            0.5 * np.dot( np.dot((x-mean),covarInv), (x-mean))
        return lpdfval
        
    def set_log_model(self):        
        for y in self.states:
            self.logA[y] = {}
            for y1 in self.A[y].keys():
                self.logA[y][y1] = math.log(self.A[y][y1])
            self.logB[y] = {}
            for sym in self.B[y].keys():
                if self.B[y][sym] == 0:
                    self.logB[y][sym] =  sys.float_info.min # this is to handle symbols that never appear in the dataset
                else:
                    self.logB[y][sym] = math.log(self.B[y][sym])
            if self.pi[y] == 0:
                self.logpi[y] =  sys.float_info.min # this is to handle symbols that never appear in the dataset
            else:
                self.logpi[y] = math.log(self.pi[y])

    def viterbi_log(self, obs):
        vit = [{}]
        path = {}
        logpi, logA = np.log(self.pi), np.log(self.A)
        # Initialize base cases (t == 0)
        for y in self.states:
            vit[0][y] = logpi[y] + self._lpdf(obs[0],self.means[y],self.covarsInv[y])
            path[y] = [y]
     
        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}     
            for y in self.states:
                (prob, state) = max((vit[t-1][y0] + logA[y0][y] + \
                    self._lpdf(obs[t],self.means[y],self.covarsInv[y]), y0)\
                    for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]
            # Don't need to remember the old paths
            path = newpath
        n = 0           # if only one element is observed max is sought in the initialization values
        if len(obs)!=1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])

    def viterbi_k_log(self,obs,k):
        vit = [{}]
        path = {}
        logpi, logA = np.log(self.pi), np.log(self.A)
        for y in self.states:
            vit[0][y] = [logpi[y] + self._lpdf(obs[0],self.means[y],self.covarsInv[y])]
            path[y] = [[y]]
        
        for t in range(1,len(obs)):
            vit.append({})
            newpath = {}
            for y in self.states:
                newheap = []
                for y0 in self.states:
                    for i in range(len(vit[t-1][y0])):
                        v, p = vit[t-1][y0][i], path[y0][i]
                        if len(newheap)<k:
                            heapq.heappush(newheap,(v+logA[y0][y]+self._lpdf(obs[t], \
                                        self.means[y],self.covarsInv[y]),p+[y]))
                        else:
                            heapq.heappushpop(newheap,(v+logA[y0][y]+self._lpdf(obs[t],\
                                        self.means[y],self.covarsInv[y]),p+[y]))
                vit[t][y] = [x[0] for x in newheap]
                newpath[y] = [x[1] for x in newheap]
            path = newpath
        n, res = 0, []
        if len(obs)!=1:
            n = t
        for y in self.states:
            for i in range(len(vit[n][y])):
                if len(res)<k:
                    heapq.heappush(res,(vit[n][y][i],path[y][i]))
                else:
                    heapq.heappushpop(res,(vit[n][y][i],path[y][i]))
        return res

    def backward_scaled(self, obs):
        # uses the clist created during forward_scaled function
        # This assumes that forward_scaled is already execued and clist is set up properly
        
        self.bwk = [{} for t in range(len(obs))]
        self.bwk_scaled = [{} for t in range(len(obs))]
        
        T = len(obs)
        # Initialize base cases (t == T)
        for y in self.states:
            self.bwk[T-1][y] = 1 #self.A[y]["Final"] #self.pi[y] * self.B[y][obs[0]]
            try:
                self.bwk_scaled[T-1][y] = self.clist[T-1] * 1.0 #
            except:
                print "EXCEPTION OCCURED in backward_scaled, T -1 = ", T -1
        
        for t in reversed(range(T-1)):
            beta_local = {}
            for y in self.states:
                beta_local[y] = sum((self.bwk_scaled[t+1][y1] * self.A[y][y1] * self._pdf(obs[t+1],self.means[y1],self.covarsInv[y1])) for y1 in self.states)
                
            for y in self.states:
                self.bwk_scaled[t][y] = self.clist[t] * beta_local[y]
        
        log_p = -sum([math.log(c) for c in self.clist])
        
        # NOTE: if log probabilty is very low, prob can turn out to be zero
        #prob = math.exp(log_p) 
        return log_p # prob    

    # compute c values given the pointer to alpha values
    def compute_cvalue(self, alpha, states):
        alpha_sum = 0.0
        for y in states:
            alpha_sum += alpha[y]
        if alpha_sum == 0:
            # given that the initial prob in the base case at least is non zero we dont expect alpha_sum to become zero
            print "Critical Error, sum of alpha values is zero"
        cval = 1.0 / alpha_sum
        if cval == 0:
            print "ERROR cval is zero, alpha = ", alpha_sum
        return cval

    # this function implements the forward algorithm from Rabiner's paper
    # this implements scaling as per the paper and the errata
    # given an observation sequence (a list of symbols) and Model, compute P(O|Model)
    def forward_scaled(self, obs):
        self.fwd = [{}]
        local_alpha = {} # this is the alpha double caret in Rabiner
        self.clist = [] # list of constants used for scaling
        self.fwd_scaled = [{}] # fwd_scaled is the variable alpha_caret in Rabiner book
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self._pdf(obs[0],self.means[y],self.covarsInv[y])

        # get c1 for base case
        c1 = self.compute_cvalue(self.fwd[0], self.states)
        self.clist.append(c1)
        # create scaled alpha values
        for y in self.states:
            self.fwd_scaled[0][y] = c1 * self.fwd[0][y]
            
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd.append({})
            self.fwd_scaled.append({})
            for y in self.states:
                #self.fwd[t][y] = sum((self.fwd[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
                accm = 0.0
                for y0 in self.states:
                    pval = (self.fwd_scaled[t-1][y0] * \
                               self.A[y0][y] * self._pdf(obs[t],self.means[y],\
                                self.covarsInv[y]))
                    accm+=pval
#                    if accm==0:
#                        print self.fwd_scaled[t-1][y0], self._pdf(obs[t],self.means[y],\
#                                self.covarsInv[y]), obs[t],self.means[y],\
#                                self.covarsInv[y]
                local_alpha[y] = max(accm,1e-20)
                if (local_alpha[y] == 0):
                    raise Exception("ERROR local alpha is zero: y = ", y,self.fwd_scaled[t-1]) #, "  y0 = ", y0
#                        print "fwd = %3f, A = %3f, B = %3f, obs = %s" % (self.fwd_scaled[t - 1][y0], self.A[y0][y], self.B[y][obs[t]], obs[t])

            c1 = self.compute_cvalue(local_alpha, self.states)
            self.clist.append(c1)
            # create scaled alpha values
            for y in self.states:
                self.fwd_scaled[t][y] = c1 * local_alpha[y]

        log_p = -sum([math.log(c) for c in self.clist])
        
        # NOTE: if log probabilty is very low, prob can turn out to be zero
        #prob = math.exp(log_p) #sum((self.fwd[len(obs) - 1][s]) for s in self.states)        
        return log_p # prob    

    def viterbi(self, obs):
        vit = [{}]
        path = {}     
        # Initialize base cases (t == 0)
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]
     
        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}     
            for y in self.states:
                (prob, state) = max((vit[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]
            # Don't need to remember the old paths
            path = newpath
        n = 0           # if only one element is observed max is sought in the initialization values
        if len(obs)!=1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])

    def forward_backward_multi_scaled(self, obslist, thresh): # returns model given the initial model and observations
        count = 40
        def eachInv(x):
            return np.matrix(x).I
        log_p = -100000.0
        for iteration in range(count):
            tables = self.create_xi_gamma_tables(obslist)            
            # compute transition probability
            temp_aij = np.zeros(self.A.shape)
            temp_means = np.zeros(self.means.shape)
            temp_cov = np.zeros(self.covars.shape)
            temp_pi = np.zeros(self.N)

            for i in self.states:
                temp_pi[i] = self.compute_pi(tables, i)
                temp_means[i], temp_cov[i] = self.compute_dist(tables,i,obslist)
                for j in self.states:
                    temp_aij[i][j] = self.compute_aij(tables, i, j)
            temp_pi = temp_pi/sum(temp_pi)

            self.A = temp_aij
            self.means = temp_means
            self.covars = temp_cov
            self.covarsInv = np.array([eachInv(x) for x in self.covars])
            self.pi = temp_pi
            log_p1 = -sum([math.log(c) for c in self.clist])
            print "Iter: ", iteration, "; Current Log-Likelihood: ", log_p1
            if (log_p1-log_p)<thresh:
                print "Converged! Existing..."
                break
            log_p = log_p1
        return (temp_aij, temp_means, temp_cov, temp_pi)

    # compute aij for a given (i, j) pair of states
    def compute_aij(self, tables, i, j):
        xi_table = tables["xi_table"] # this will have xi values [k][t][i][j]
        gamma_table = tables["gamma_table"] # this will have gamma values [k][t][i]
        numerator = 0.0
        denominator = 0.0
        
        for k in range(len(xi_table)): # sum over all observations in the multi list
            for t in range(len(xi_table[k]) - 1): # sum over all t up to Tk - 1
                denominator += gamma_table[k][t][i] # xi value for i, j
                numerator += xi_table[k][t][i][j] # xi value for i, j
        aij = numerator / denominator
        return aij

    # compute the emission probabilities of a given state i emitting symbol
    def compute_dist(self, tables, i, obslist):
#        threshold = 0 # TODO: support for setting some minimum value if bj turns out to be 0 - we also need to ensure probabilities sum to 1
        gamma_table = tables["gamma_table"] # this will have gamma values [k][t][i]
        numerator_mean =  np.zeros(self.d)
        numerator_cov = np.zeros(self.covars[0].shape)
        cov_prior = np.eye(self.d)*self.min_std
        denominator = 0.0
        for k in range(len(gamma_table)):
            for t in range(len(gamma_table[k])-1):
                denominator += gamma_table[k][t][i]
                numerator_mean += gamma_table[k][t][i] * obslist[k][t]
                numerator_cov += gamma_table[k][t][i] * \
                    np.outer(obslist[k][t]-self.means[i],obslist[k][t]-self.means[i])
        mean_new = numerator_mean/denominator
        covar_new = numerator_cov/denominator+cov_prior
        return mean_new, covar_new

    # compute the initial probabilities of a given state i 
    def compute_pi(self, tables, i):
        gamma_table = tables["gamma_table"] # this will have gamma values [k][t][i]

        pi = 0.0
        for k in range(len(gamma_table)): # sum over all observations in the multi list
            pi += gamma_table[k][0][i] #xi_table[k][t][i][j] # xi value for i, j
        return pi

    def compute_xi(self, alphas, betas, qi, qj, obs):
        # given alpha and beta tables and the states qi, qj, computes xi values, assumes A, B, pi are available
        xi = alphas[qi] * self.A[qi][qj] * self._pdf(obs,self.means[qj],self.covarsInv[qj]) * betas[qj]
        return xi
        
    def compute_gamma(self, alphas, betas, qi, ct):
        # given alpha and beta tables and the states qi, qj, computes xi values, assumes A, B, pi are available
        gam = (alphas[qi] * betas[qi]) / float(ct)
        if gam == 0:
            # TODO: Handle any error situation arising due to gamma = 0
            #print "gam = ", gam, " alpha = ", alphas[qi], " beta = ", betas[qi], " qi = ", qi
            pass
        return gam

    def create_xi_gamma_tables(self, obslist):
        # we will create a table for xi that stores xi(i, j) for all t and all k, all i and j
        # also create a table for gamma that stores gamma(i) for all t and all k, all i
        # where t is the sequence index: 1 <= t <= Tk - 1
        # and k is the observation number: 1 <= k <= K
        
        xi_table = [] # each element in this is for a given obs in obslist, obs is a vector of symbols
        gamma_table = [] # each element in this is for a given obs in obslist, obs is a vector of symbols
        
        for obs in obslist: # do for every observation sequence from the multi observation list
            # first create the scaled alpha and beta tables by calling self.forward_scaled and self.bakward_scaled
            # these will set up scaled alpha and beta tables properly for the given observation sequence
            self.forward_scaled(obs)
            self.backward_scaled(obs)
            #xi_obs = [] # this holds the xi for kth observation
            xi_t = [] # this holds the xi for Tk - 1
            gamma_t = [] # this holds the gamma for Tk - 1

            for t in range(len(obs) - 1): # 1 <= t <= Tk - 1
                xi_t.append({}) # this holds xi for the given k and t - it should have (i, j) entries
                gamma_t.append({}) # this holds gamma for the given k and t - it should have i entries
                for i in self.states:
                    xi_t[t][i] = {}
                    gamma_t[t][i] = self.compute_gamma(self.fwd_scaled[t], self.bwk_scaled[t], i, self.clist[t])
                    for j in self.states:
                        xi_t[t][i][j] = self.compute_xi(self.fwd_scaled[t], self.bwk_scaled[t + 1], i, j, obs[t + 1])
            xi_table.append(xi_t)
            gamma_table.append(gamma_t)
        return {"xi_table": xi_table, "gamma_table": gamma_table}

#if __name__ == '__main__':
    
    
    
