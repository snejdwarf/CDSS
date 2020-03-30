#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import math
from scipy import optimize

import src.dataparser as dp
from src.quickscore import Quickscore
from src import util

import time


# In[ ]:


class VI:

    def __init__(self):
        self.iterationCounter = -9001
    
    def theta_ij(self, i, j, Q):
        return -math.log(1-Q[i,j])
    
    def theta_izero(self):
        smallprob = 1e-5
        return -math.log(1-smallprob)
    
    def conjugate(self, xi):
        assert xi>0, "xi <= 0"
        return -xi*math.log(xi)+(xi+1)*math.log(xi+1)
    
    def optimize_parameters(self, trans_findings, exact_findings,qs, all_posts = None, _xi = None, useGradient=False,useFunction=True):
        '''Function that takes findings to transform and those to handle exact and returns the XI-parameters'''

        self.iterationCounter = 0

        ND = qs.N_disease()
        NS = qs.N_findings()
        if all_posts == None:
            all_posts = qs.mple_dis_post_fast([i for i in range(ND)],exact_findings,[])
        if _xi is not None:
            local_XI = _xi
        else:
            local_XI = [2 for i in range(NS)]

        obj_int = None

        if useGradient==2:
            func = self.logP_fplus_eps_improved2 if useFunction else None
            obj_int = obj=optimize.minimize(func,local_XI,args=(trans_findings,exact_findings,qs,all_posts),bounds=[(0.0001,None) for i in range(NS)],callback = self.callback1, jac=self.all_gradients)
            local_XI = obj.x
            print(f"---intermediate---\n"
                  f"fun: {obj.fun}\n"
                  f"jac: {obj.jac[:6]}\n"
                  f"msg: {obj.message}\n"
                  f"nfev: {obj.nfev}\n"
                  f"nit: {obj.nit}\n"
                  f"status: {obj.status}\n"
                  f"success: {obj.success}\n"
                  f"x: {obj.x[:6]}\n"
                  f"---\n")

            if not useFunction:
                print("Continuing with function even though useFunction is False")
            obj=optimize.minimize(self.logP_fplus_eps_improved2,local_XI,args=(trans_findings,exact_findings,qs,all_posts),bounds=[(0.0001,None) for i in range(NS)],callback = self.callback1, jac=None)
        else:
            gradient = None if useGradient is False else self.all_gradients
            obj=optimize.minimize(self.logP_fplus_eps_improved2,local_XI,args=(trans_findings,exact_findings,qs,all_posts),bounds=[(0.0001,None) for i in range(NS)],callback = self.callback1, jac=gradient)

        print(f"---\n"
              f"fun: {obj.fun}\n"
              f"jac: {obj.jac[:6]}\n"
              f"msg: {obj.message}\n"
              f"nfev: {obj.nfev}\n"
              f"nit: {obj.nit}\n"
              f"status: {obj.status}\n"
              f"success: {obj.success}\n"
              f"x: {obj.x[:6]}\n"
              f"---\n")
        return obj_int, obj

    def delta_ordering(self, pos_finds, qs):
        '''Function that returns the findings in the order that uses the delta heuristic to find the order to introduce the findings in'''
        xi_all_trans = self.optimize_parameters(pos_finds,[],qs).x
        all_trans_res = self.P_fplus_eps(xi_all_trans,pos_finds,[],qs)

        #dictionary to hold difference
        findingMapdiff = {}

        #Calculate differences that arise from transforming the i'th finding vs not transforming it
        for i in pos_finds:
            iter_findings = list(pos_finds)
            iter_findings.remove(i)
            i_removed_res = self.P_fplus_eps(xi_all_trans,iter_findings,[i],qs)
            diff = all_trans_res-i_removed_res
            findingMapdiff[i] = diff


        sorted_findings = [f for f,v in sorted(findingMapdiff.items(), key = lambda x1: x1[1], reverse=True)]
        return sorted_findings, xi_all_trans

    def callback1(self, XI):
        self.iterationCounter += 1
        print("iteration %d" % self.iterationCounter, XI[:4])

    def updatePriors(self, XI, transformed_findings, PD):
        #model = self.model

        NF, ND = Q.shape

        theta = self.theta_ij

        PD = PD.copy()

        for i, f in enumerate(transformed_findings):
            for d in range(ND):
                PD[d] *= np.exp(XI[f] * theta(f, d, Q))
                #model.PD[d] = (np.exp(XI[i] * theta(f, d)) * model.PD[d]) \
                #            / (np.exp(XI[i] * theta(f, d)) * model.PD[d] + (1-model.PD[d]))

                PD[d] = max(1e-12, PD[d])

        return PD

    def all_gradients(self, XI, trans_findings,exact_findings,qs,all_posts):
        # TODO: check xi parameters. Think this class has one for all findings?
        #  This seems to be what is causing the error: this only returns xi gradients
        #  for the transformed findings

        # TODO: gradient w/ updates priors?

        newPD = PD

        newPD = self.updatePriors(XI, trans_findings, PD)

        qs = Quickscore.QS(Q, newPD)
        newPD = np.array(list(qs.mple_dis_post_fast(range(Q.shape[1]), exact_findings, ()).values()))
        #print(repr(newPD))

        log_gradient = self.log_gradient

        #grad = np.empty(Q.shape[0])
        grad = np.full(Q.shape[0], 0.0)
        for i, f in enumerate(trans_findings):
            grad[f] = log_gradient(XI, i, f, newPD)#*1000

        return grad

    def log_gradient(self, XI, i, f, PD):
        from math import log
        #ND = qs.ND
        #PD = qs.PD
        ND = Q.shape[1]
        theta_i0 = lambda x : 0  # self.theta_i0
        theta = self.theta_ij

        #assert XI[i] > 0.0, "xi_%d > 0 for finding %d = %f" % (i,f, XI[i])
        xi = XI[f]
        if xi <= 0.0:
            xi = 10e-12

        return theta_i0(f) + log(xi/(1+xi))                + sum(( (theta(f, j, Q=Q) * PD[j]) for j in range(ND)))
        
    def logP_fplus_eps_improved2(self, XI,trans_findings,exact_findings,qs,all_posts_arg = None):
        '''Function to calculate the log probability of the positive findings with some treated exact and some treatet approximately.
            Implements eq. 44 from paper.
        '''
        ND = qs.N_disease()
        # Calculate first term
        term1 = 0
        for i in trans_findings:
            term1 = term1 + (- self.conjugate(XI[i]) ) #Background removed

        # Calculate second term: log to the expectation value
        term2 = 1
        if all_posts_arg == None:
            all_posts = qs.mple_dis_post_fast([i for i in range(ND)],exact_findings,[])
        else:
            all_posts = all_posts_arg

        for j in range(ND):

            # Calculate P(d_j|f+) using quickscore
            post_dj = all_posts[j]

            # Calculate d_j = 0 term
            term_dj0 = 1-post_dj

            # Calculate dj = 1 term
            exp_sum = 0
            for i in trans_findings:
                exp_sum = exp_sum + XI[i]*self.theta_ij(i,j,qs.Q)
            term_dj1 = post_dj*np.exp(exp_sum)

            term2 = term2 * (term_dj0+term_dj1)
        term2 = np.log(term2)

        return term1 + term2
    
    
    def P_fplus_eps(self, XI,trans_findings,exact_findings,qs,all_posts_arg = None):
        '''Function to calculate the probability of the positive findings with some treated exact and some treatet approximately.
            Implements eq. 44 from paper and takes the exponent. --- Constant (likelihood of exactly treated findings) is included here.
        '''

        ND = qs.N_disease()

        # Calculate first term
        term1 = 0
        for i in trans_findings:
            term1 = term1 + (- self.conjugate(XI[i]) )

        # Calculate second term: log to the expectation value
        term2 = 1
        if all_posts_arg == None:
            all_posts = qs.mple_dis_post_fast([i for i in range(ND)],exact_findings,[])
        else:
            all_posts = all_posts_arg
        print('posts calculated')

        for j in range(ND):

            # Calculate P(d_j|f+) by lookup
            post_dj = all_posts[j]

            # Calculate d_j = 0 term
            term_dj0 = 1-post_dj

            # Calculate dj = 1 term
            exp_sum = 0
            for i in trans_findings:
                exp_sum = exp_sum + XI[i]*self.theta_ij(i,j,qs.Q)
            term_dj1 = post_dj*np.exp(exp_sum)

            term2 = term2 * (term_dj0+term_dj1)
        term2 = np.log(term2)

        # Calculate constant C: the likelihood of the exactly treated findings 
        qs_res=qs.PFindings(exact_findings,[])
        #print("qs_result: ", qs_res)
        C = np.log(qs_res)

        sumterms = term1 + term2 + C
        res = np.exp(sumterms)
        #print(res)

        return res
    
    ### Everything lower bound related - start

    def f(self, x):
        '''f is the function introduced in the paper'''
        return np.log(1-np.exp(-x))
    
    def g(self, i,j, lower_bound_q,Q):
        return self.f(self.theta_izero()+(self.theta_ij(i,j,Q)/lower_bound_q))-self.f(self.theta_izero()) #Background included here

    def f_numericGradient(self, x):

        smallNumb = np.float(1e-8)
        y1 = self.f(x)
        y2 = self.f(x+smallNumb)
        return (y2-y1)/smallNumb
    
    def f_analyticGradient(self, x):
        e = np.exp(-x)
        return e/(1-e)

    def lowerBoundJoinFindings(self,Q,PD,transformed_findings, exact_positive, exact_negative, q_params):
        '''
        Function that calculates the lower bound on the joint probability of the findings for the given q-parameters.
        '''
        PD_copy = np.array(PD)
        qs = Quickscore.QS(Q,PD_copy,onlyupdatepd=True)
        background = 1
        for i in transformed_findings:
            thetazero = self.theta_izero()
            f_thetazero = self.f(thetazero)
            d_parents_i = util.findingRelatedDis(Q,i)
            for j in d_parents_i:
                q_ij = q_params[i,j]
                theta_ij = self.theta_ij(i,j,Q)
                g = self.f(thetazero + (theta_ij/q_ij))-f_thetazero 
                qs.PD[j] *= np.exp(q_ij*g)
                #print(np.exp(q_ij*g),' arg for f: ', theta_ij/q_ij, ' theta: ', theta_ij, ' q_param: ', q_ij, ' div: ', theta_ij/q_ij, ' function: ', g)
            background *= np.exp(f_thetazero)
        
        #print(qs.PD - PD)
        
        return qs.PFindings(exact_positive,exact_negative)*background
    
    def posteriorLowerBoundApproximations(self, Q, PD, transformed_findings, exact_positive, exact_negative, q_params):
        '''Function that absorbs the transformed findings given the lower bound q-parameters and returns the lower bound posteriors p(d_j|f+,q)'''
        PD_copy = np.array(PD)
        
        # Absorb transformed findings
        for i in transformed_findings:
            d_parents_i = util.findingRelatedDis(Q,i)
            for j in d_parents_i:
                qlb_ij = q_params[i,j]
                g_ij = self.g(i,j,qlb_ij,Q)
                nom = np.exp(qlb_ij*g_ij)*PD_copy[j]
                den = nom + (1-PD_copy[j])
                PD_copy[j] = nom/den
                
        qs_local = Quickscore.QS(Q,PD_copy)
        return qs_local.mple_dis_post_fast_v2([i for i in range(len(PD))],exact_positive,exact_negative)

       
       
    def updateQlbParameters_oneIteration(self, Q, PD, transformed_findings, exact_positive, exact_negative, q_params_old):
        q_res = np.array(q_params_old)
        n_rows, n_cols = q_res.shape
        
        #Calculate expectation for each d_j
        posterior_approx = self.posteriorLowerBoundApproximations(Q,PD,transformed_findings,exact_positive,exact_negative, q_params_old)
        for i in transformed_findings:
            for j in util.findingRelatedDis(Q,i):
                exp_dj = posterior_approx[j[0]]
                q_ij = q_params_old[i,j]
                theta_i0 = self.theta_izero()
                theta_ij = self.theta_ij(i,j,Q)
                
                f_arg = theta_i0+(theta_ij/q_ij)
                
                f = self.f(f_arg)
                f_derivative = self.f_analyticGradient(f_arg)
                
                q_res[i,j] = exp_dj * ( q_ij*f - theta_ij*f_derivative - q_ij*self.f(theta_i0) )
        return q_res
    
    def updateQlbParameters(self, Q, PD, transformed_findings, exact_positive, exact_negative, q_params_old=None,maxIter=200):
        if q_params_old is None:
            q_params_old = np.ones(Q.shape)*np.random.random()
        
        q_sumold = 0
        q_updated = q_params_old
        for i in range(maxIter):
            q_updated=self.updateQlbParameters_oneIteration(Q,PD,transformed_findings,exact_positive,exact_negative,q_updated)
            #Normalize parameters
            q_updated=util.rowNormalize2DArray(q_updated)
            #Variable to keep track of convergence
            q_sum=0
            for ii in transformed_findings:
                for j in util.findingRelatedDis(Q,ii):
                    q_sum += q_updated[ii,j]

            print("q_sum diff: ", abs(q_sumold-q_sum))
            # Break out of loop if the sum q parameters doesn't change significantly
            if abs(q_sumold-q_sum) < 1e-5:
                print(i)
                break
            q_sumold = q_sum
        return q_updated
    
    
    def absorbNegativeFindings(self, Q, PD, negative_trans, negative_rest):
        
        PD_copy = np.array(PD)
        qs = Quickscore.QS(Q,PD_copy, onlyupdatepd=True)
        for i in negative_trans:
            for j in util.findingRelatedDis(Q,i):
                qs.PD[j] *= (1-Q[i,j])
                pass;
        return qs.PFindings([],negative_rest)

    ### Lower bound -  end

