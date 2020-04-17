#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import math
from scipy import optimize

from src.quickscore import Quickscore
from src import util
import time
import  numbers

import time


# In[ ]:


class VI:

    def __init__(self):
        self.iterationCounter = -9001
    

    
    def theta_izero(self):
        smallprob = 1e-5
        return -math.log(1-smallprob)
    
    def conjugate(self, xi):
        #if xi<=0 or np.isnan(xi):
        #    return 0
        assert xi>0, "xi <= 0, {} ".format(xi)
        return -xi*math.log(xi)+(xi+1)*math.log(xi+1)

    def joint_probability_upper_bound(self, Q, PD, positive_findings, n_trans_findings, use_delta_ordering=False):

        qs = Quickscore.Quickscore(Q, PD)

        ordered_findings = []
        if use_delta_ordering:
            ordered_findings,xi_all_trans,_ = self.delta_ordering_ub_based(positive_findings, qs)
            exact_findings = ordered_findings[:-n_trans_findings]
            trans_findings = ordered_findings[-n_trans_findings:]
        else:
            exact_findings = positive_findings[:-n_trans_findings]
            trans_findings = positive_findings[-n_trans_findings:]
        #print("Time for delta ordering: ", tac-tic)
        #print("Trans_finds: ", trans_findings)
        if use_delta_ordering:
            xialltrans = []
            for i,f in enumerate(positive_findings):
                if f in trans_findings:
                    xialltrans.append(xi_all_trans[i])

        xi_obj,exactfindings_prob = self.optimize_parameters(trans_findings,exact_findings,qs)#xi=xialltrans if use_delta_ordering else None)
        xi = xi_obj.x
        vi_part = xi_obj.fun

        upper_bound_res = np.exp(vi_part)*exactfindings_prob

        return upper_bound_res, xi_obj, ordered_findings

    def optimize_parameters(self, trans_findings, exact_findings, qs, xi=None):
        self.iterationCounter = 0
        nd = qs.N_disease()
        ns = qs.N_findings()
        local_xi = [2 for i in range(len(trans_findings))] if xi is None else xi
        tic = time.time()
        all_posts,finding_prob = qs.mple_dis_post_fast_v3(util.parentsOfFindings(qs.Q,trans_findings), exact_findings, [],return_finding_prob=True)
        tac = time.time()
        func = self.logP_fplus_eps
        obj = optimize.minimize(func, local_xi, args=(trans_findings, exact_findings, qs, all_posts),
                                bounds=[(0.0001, None) for i in range(len(trans_findings))],callback=self.callback)

        return obj, finding_prob


    def delta_ordering_ub_based(self, pos_finds, qs):
        '''Function that returns the findings in the order that uses the delta heuristic to find the order to introduce the findings in'''
        xi_all_trans = self.optimize_parameters(pos_finds,[],qs)[0].x
        all_trans_res = self.P_fplus_eps(xi_all_trans,pos_finds,[],qs)
        #dictionary to hold difference
        findingMapdiff = {}

        #Calculate differences that arise from transforming the i'th finding vs not transforming it
        for index,i in enumerate(pos_finds):
            iter_findings = list(pos_finds)
            iter_xi = list(xi_all_trans)

            iter_findings.remove(i)
            del iter_xi[index]

            i_removed_res = self.P_fplus_eps(iter_xi,iter_findings,[i],qs)
            diff = all_trans_res-i_removed_res
            findingMapdiff[i] = diff

        # Sort by the ones that are best to treat exactly
        sorted_findings = [f for f,v in sorted(findingMapdiff.items(), key=lambda x1: x1[1], reverse=True)]
        return sorted_findings, xi_all_trans, findingMapdiff

    def delta_ordering_lb_based(self, pos_finds, qs):
        Q = qs.Q
        PD = qs.PD
        q_parameters_all_trans = self.update_qlb_parameters(Q, PD, pos_finds, [], [], maxIter=200)
        all_trans_res = self.joint_probability_lowerbound_givenparameters(Q, PD, pos_finds, [], [], q_parameters_all_trans)
        findingMapdiff= {}

        for index,i in enumerate(pos_finds):
            iter_findings = list(pos_finds)
            iter_q_parameters = list(q_parameters_all_trans)

            iter_findings.remove(i)
            del iter_q_parameters[index]

            i_removed_res = self.joint_probability_lowerbound_givenparameters(Q, PD, iter_findings, [i], [], iter_q_parameters)
            diff = abs(all_trans_res-i_removed_res)
            findingMapdiff[i] = diff

        sorted_findings = [f for f,v in sorted(findingMapdiff.items(), key = lambda x1: x1[1], reverse=True)]
        return sorted_findings, q_parameters_all_trans, findingMapdiff

    def callback(self, XI):
        self.iterationCounter += 1

    def logP_fplus_eps(self, XI, trans_findings, exact_findings, qs, all_posts_arg = None):
        '''Function to calculate the log probability of the positive findings with some treated exact and some treatet approximately.
        '''
        ND = qs.N_disease()
        # Calculate first term
        #term1 = 0

        term1 = 0
        for xi in XI:
            term1 = term1 + (- self.conjugate(xi))

        # Calculate second term: log to the expectation value
        term2 = 1
        if all_posts_arg == None:
            relevantFindings = util.parentsOfFindings(qs.Q,trans_findings)
            all_posts = qs.mple_dis_post_fast_v3(relevantFindings,exact_findings,[])
        else:
            all_posts = all_posts_arg


        for xi_index,i in enumerate(trans_findings):

            for j in util.findingRelatedDis(qs.Q,i):

                post_dj = all_posts[j]

                term2 *= (1 - post_dj) + post_dj*np.exp(XI[xi_index]*self.theta_ij(i,j,qs.Q))

        term2 = np.log(term2)

        return term1 + term2


    def P_fplus_eps(self, XI, trans_findings, exact_findings, qs, all_posts_arg = None):
        '''Function to calculate the probability of the positive findings with some treated exact and some treatet approximately.
             --- Constant (likelihood of exactly treated findings) is included here.
        '''

        nd = qs.N_disease()

        # Calculate first term
        term1 = 0
        for xi in XI:
            term1 = term1 + (- self.conjugate(xi))

        # Calculate second term: log to the expectation value
        term2 = 1
        if all_posts_arg == None:
            all_posts = qs.mple_dis_post_fast_v3([i for i in range(nd)],exact_findings,[])
        else:
            all_posts = all_posts_arg

        for xi_index,i in enumerate(trans_findings):

            for j in util.findingRelatedDis(qs.Q,i):

                post_dj = all_posts[j]

                term2 *= (1 - post_dj) + post_dj*np.exp(XI[xi_index]*self.theta_ij(i,j,qs.Q))

        term2 = np.log(term2)

        # Calculate constant C: the likelihood of the exactly treated findings
        qs_res=qs.probability_of_findings(exact_findings, [])
        C = np.log(qs_res)

        sumterms = term1 + term2 + C
        res = np.exp(sumterms)

        return res
    
    ### Everything lower bound related - start

    def theta_ij(self, i, j, Q):
        return -math.log(1-Q[i,j])

    def f(self, x):
        '''f is the function introduced in the paper'''
        return np.log(1-np.exp(-x))
    
    def g(self, i,j, lower_bound_q,Q):
        if lower_bound_q == 0:
            return 0

        res = self.f(self.theta_izero()+self.theta_ij(i,j,Q)/lower_bound_q)-self.f(self.theta_izero())

        if(res < -1e10):
            print("Parts of res:", (self.theta_ij(i,j,Q)/lower_bound_q ))
        return res

    def f_numericGradient(self, x):

        smallNumb = np.float(1e-8)
        y1 = self.f(x)
        y2 = self.f(x+smallNumb)
        return (y2-y1)/smallNumb
    
    def f_analyticGradient(self, x):
        e = np.exp(-x)
        return e/(1-e)

    def joint_probability_lowerbound_givenparameters(self, Q, PD, transformed_findings, exact_positive, exact_negative, q_params):
        '''
        Function that calculates the lower bound on the joint probability of the findings for the given q-parameters. Splits result into two factors: one for the exact probability and one for the transformed
        '''

        qs = Quickscore.Quickscore(Q, PD, onlyupdatepd=True)
        relevant_posteriors,finding_prob = qs.mple_dis_post_fast_v3(list(range(len(PD))),exact_positive,[],return_finding_prob=True)
        #print("finding_prob", finding_prob)

        relevant_diseases = list(relevant_posteriors.keys())
        #Calculate factor to be multiplied on the exact result
        factor = 1
        for j in relevant_diseases:
            post_dj = relevant_posteriors[j]
            #calculate exponent
            exponent = 0
            for i in transformed_findings:
                q = q_params[i][j]
                g = self.g(i,j,q,Q)
                exponent += q*g

            factor *= (np.exp(exponent)*post_dj + (1-post_dj))

        #background
        back = 1
        for i in transformed_findings:
            back *= np.exp(self.f(self.theta_izero()))

        return finding_prob * factor * back
    
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
                
        qs_local = Quickscore.Quickscore(Q, PD_copy)
        return qs_local.mple_dis_post_fast_v3([i for i in range(len(PD))],exact_positive,exact_negative)


       
    def updateQlbParameters_oneIteration(self, Q, PD, transformed_findings, exact_positive, exact_negative, q_params_old, relevant_posteriors):
        q_res = np.array(q_params_old)

        for i in transformed_findings:
            for j in util.findingRelatedDis(Q,i):
                #Calculate expectation
                    #exponent
                exp=0

                for ii in transformed_findings:
                    q = q_params_old[ii,j]
                    if q <= 1e-14 or np.isnan(q):
                        continue
                    if Q[ii,j] > 0:
                        g = self.g(ii,j,q,Q)
                    else:
                        g = 0
                    exp += q*g
                    assert not np.isnan(g), 'nan in exponent detected'

                expect_num = np.exp(exp)*relevant_posteriors[j]
                expect_den = expect_num + (1-relevant_posteriors[j])
                assert expect_den > 0, 'Denominator less that or equal to zero'
                assert not np.isnan(expect_den), 'nan detected'
                expect = expect_num/expect_den

                #Calculate factor2 to be multiplied on expectation:
                q = q_params_old[i,j]
                thetai0 = self.theta_izero()
                thetaij = self.theta_ij(i,j,Q)
                assert q>0, 'lower bound q parameter less than or equal to zero'
                f = self.f(thetai0+thetaij/q)
                f_derivative = self.f_analyticGradient(thetai0+thetaij/q)
                f_i0 = self.f(thetai0)
                factor2 = q*f-thetaij*f_derivative-q*f_i0
                updated_q = expect * factor2
                q_res[i,j] = max(1e-15,updated_q)

        return q_res
    
    def update_qlb_parameters(self, Q, PD, transformed_findings, exact_positive, exact_negative, q_params_old=None, maxIter=200, change_limit=1e-7):
        if q_params_old is None:
            #initialize q-parameters
            #q_params_old = np.zeros((len(transformed_findings), Q.shape[1]))
            #q_params_old = np.zeros(Q.shape)
            q_params_old = np.ones(Q.shape)

            #for i in transformed_findings:
            #    for j in util.findingRelatedDis(Q,i):
            #        q_params_old[i,j] = 1

        relevant_diseases = util.parentsOfFindings(Q,transformed_findings)
        qs = Quickscore.Quickscore(Q, PD)
        relevant_posteriors = qs.mple_dis_post_fast_v3(relevant_diseases, exact_positive,[])

        q_sumold = 1e-10
        q_updated = q_params_old
        for i in range(maxIter):
            q_updated=self.updateQlbParameters_oneIteration(Q,PD,transformed_findings,exact_positive,exact_negative,q_updated, relevant_posteriors)
            #Normalize parameters
            q_updated=util.rowNormalize2DArray(q_updated)
            #Variable to keep track of convergence
            q_sum=0
            for ii in transformed_findings:
                for j in util.findingRelatedDis(Q,ii):
                    q_sum += q_updated[ii,j]

            #print("q_sum diff: ", abs(q_sumold-q_sum))
            # Break out of loop if the sum q parameters doesn't change significantly
            if q_sum/q_sumold-1 < change_limit:
                print(i)
                break
            q_sumold = q_sum
        if i==maxIter-1:
            print(i)
        return q_updated

    def joint_probability_lower_bound(self, Q, PD, positive_findings, n_trans_findings, use_delta_ordering_lbb=False, use_delta_ordering_ubb=False, findings_ordered = None):

        if use_delta_ordering_lbb and use_delta_ordering_ubb:
            raise Exception("Can only use one type of delta ordering")

        qs = Quickscore.Quickscore(Q, PD, onlyupdatepd=True)

        # in case of no sorting
        best_findings_to_keep = positive_findings

        if findings_ordered is not None:
            use_delta_ordering_lbb=False
            use_delta_ordering_ubb=False
            best_findings_to_keep = findings_ordered

        if use_delta_ordering_ubb:
            best_findings_to_keep = self.delta_ordering_ub_based(positive_findings,qs)[0]

        if use_delta_ordering_lbb:
            best_findings_to_keep = self.delta_ordering_lb_based(positive_findings,qs)[0]



        if n_trans_findings ==0:
            trans_f = []
            exact_f = positive_findings
        elif n_trans_findings<len(positive_findings):
            trans_f = best_findings_to_keep[-n_trans_findings:]
            exact_f = best_findings_to_keep[:-n_trans_findings]
        else:
            # If the number of findings to transform is greater than the number of findings under consideration; transform all:
            trans_f = positive_findings
            exact_f = []

        #print("here: ", trans_f)
        q = self.update_qlb_parameters(Q,PD,trans_f,exact_f,[])
        return self.joint_probability_lowerbound_givenparameters(Q, PD, trans_f, exact_f, [], q)


    def joint_probability_bound(self, Q, PD, positive_findings, n_trans, use_delta_ordering = True):
        upper_bound,_,ordering = self.joint_probability_upper_bound(Q,PD,positive_findings,n_trans, use_delta_ordering=True)

        lower_bound = self.joint_probability_lower_bound(Q,PD,positive_findings,n_trans,findings_ordered=ordering)
        return lower_bound,upper_bound

    
    def absorb_negative_findings(self, Q, PD, negative_trans, negative_rest):
        
        PD_copy = np.array(PD)
        qs = Quickscore.Quickscore(Q, PD_copy, onlyupdatepd=True)
        for i in negative_trans:
            for j in util.findingRelatedDis(Q,i):
                qs.PD[j] *= (1-Q[i,j])
                pass;
        return qs.probability_of_findings([], negative_rest)

    def PFindings_factorize(self, Q, PD, positive_findings, negative_findings):
        relevant_diseases = list(range(len(PD)))
        qs = Quickscore.Quickscore(Q, PD)
        relevant_posts,finding_prob = qs.mple_dis_post_fast_v3(relevant_diseases,positive_findings,[],return_finding_prob=True)

        factor = 1

        for j in relevant_diseases:
            iter_factor = 1
            for i in negative_findings:
                iter_factor*=(1-Q[i,j])
            factor *= iter_factor*relevant_posts[j] + (1-relevant_posts[j])


        qs_pos_part = qs.probability_of_findings(positive_findings, [])


        return finding_prob*factor

 ### Lower bound -  end