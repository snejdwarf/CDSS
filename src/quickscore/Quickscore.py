#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random
import time
import sys
from src import util
import mpmath as mpm
mpm.mp.dps = 40


# In[93]:


def _powerset(s):
    if len(s) ==0:
        yield s
    else:
        for set in _powerset(s[1:]):
            yield [s[0]]+set
            yield set
            
def _powerset_tuple(s):
    if len(s) == 0:
        yield s
    else:
        for set in _powerset_tuple(s[1:]):
            yield (s[0],) + set
            yield set

class QS:

    def set_mpm_precision(self, precision):
        mpm.mp.dps = precision

    def __init__(self,Q=None,PD=None, onlyupdatepd = False):

            self.Q = Q
            self.PD = PD
            self.PD_org = np.array(PD)
            self.inverse_PD = 1-PD if PD is not None else None
            self.onlyUpdatePD = onlyupdatepd

    def convert_to_mpf(self):
        mpm_convert = np.vectorize(lambda x: mpm.mpf(x,))
        self.Q = mpm_convert(self.Q)
        self.PD = mpm_convert(self.PD)
        self.inverse_PD = mpm_convert(1-self.PD)

    def N_disease(self):
        return self.Q.shape[1]
    
    def N_findings(self):
        return self.Q.shape[0]

    def set_Q(self,Q):
        self.Q = Q
    
    def set_PD(self,PD):
        self.PD = PD

    def set_inversePD(self, invPD):
        self.inverse_PD = invPD
    
    def set_PD_org(self,PD):
        self.PD_org = PD

    def oneMinusPD(self, i):
        if self.onlyUpdatePD:
            return self.inverse_PD[i]
        else:
            return 1-self.PD[i]
        
    def generateRandomTables(self,nd,nf):
        self.PD = np.array([max(random.betavariate(2,20),1e-10) for i in range(nd)])
        self.PD_org = self.PD
        self.inverse_PD = 1-self.PD
        self.Q = np.array([[0 if random.random() > 0.1 else max(0,min(abs(random.gauss(0.03,0.6)),0.9)) for i in range(nd)] for i in range(nf)])

    def PNegativeFinding(self,negfinding):
        '''
        The probability that a single finding will be absent
        '''
        res = 1
        #For loop runs in the number of diseases
        for i in range(self.N_disease()):
            int_res = (1-self.Q[negfinding,i])*self.PD[i]+self.oneMinusPD(i)
            res = res*int_res
        return res


    def PNegativeFindings(self,*findings):
        '''
        The probability that multiple findings will be absent
        '''
        res = 1
        #Outer product
        for i in range(self.N_disease()):
            int_res = 1
            #Inner product
            for finding in findings:
                int_res = (1-self.Q[finding,i]) * int_res
            pdi = self.PD[i]
            not_pdi = self.oneMinusPD(i)
            res = res * (int_res*pdi+not_pdi)
        return res


    def PPositiveFindings(self,*findings):
        '''    
        The probability that multiple findings will be present
        '''
        findings = list(findings)
        res = 0
        for F in _powerset(findings):
            sign = (-1)**len(F)
            out_prod = 1
            for i in range(self.N_disease()):
                inn_prod = 1
                for f in F:
                    inn_prod = inn_prod*(1-self.Q[f,i])
                out_prod = out_prod * ( inn_prod*self.PD[i]+(self.oneMinusPD(i)))
            res = res + sign*out_prod
        return res

    def PFindings(self,present_findings, absent_findings, d_i = None):
        '''    
        The probability that a mixture of findings will be present or absent.
        
        If d_i is set the conditional probability of the findings given disease d_i will be returned
        '''
        res = 0
        for F in _powerset(present_findings):
            sign = (-1)**len(F)
            out_prod = 1
            for i in range(self.N_disease()):
                inn_prod = 1
                for f in F+absent_findings:
                        inn_prod = inn_prod*(1-self.Q[f,i])

                if(i==d_i):
                    out_prod = out_prod * inn_prod
                else:
                    out_prod = out_prod * ( inn_prod*self.PD[i]+(self.oneMinusPD(i)))
            res = res + sign*out_prod
        return res

    def PFindings_v2(self, present_findings, absent_findings, d_i=None, showStatus = False):
        '''
        Implements equation 11 from paper: The probability that a mixture of findings will be present or absent.
        Only iterates over relevant disease parents.
        '''
        res = 0
        iteration = 0
        for F in _powerset(present_findings):
            sign = (-1) ** len(F)
            out_prod = 1
            for i in util.parentsOfFindings(self.Q,F+absent_findings):
                inn_prod = 1
                for f in F + absent_findings:
                    inn_prod = inn_prod * (1 - self.Q[f, i])

                if (i == d_i):
                    out_prod = out_prod * inn_prod
                else:
                    out_prod = out_prod * (inn_prod * self.PD[i] + (self.oneMinusPD(i)))
            res = res + sign * out_prod
            if showStatus and iteration%round((2**len(present_findings))/8)==0:
                print(iteration/(2**len(present_findings))*100,'%')
            iteration += 1
        return res

    def PFindings_v3(self, present_findings, absent_findings, d_i=None, showStatus = False):
        '''
        The probability that a mixture of findings will be present or absent.
        Absorbs negative findings first.
        '''

        local_PD = list(self.PD)


        #Absorb evidence from the negative findings
        for i in absent_findings:
            for j in util.findingRelatedDis(self.Q,i):
                local_PD[j] *= (1-self.Q[i,j])

        res = 0
        iteration = 0
        for F in _powerset(present_findings):
            sign = (-1) ** len(F)
            out_prod = 1
            for i in util.parentsOfFindings(self.Q,F+absent_findings):
                inn_prod = 1
                for f in F:
                    inn_prod = inn_prod * (1 - self.Q[f, i])

                if (i == d_i):
                    out_prod = out_prod * inn_prod
                else:
                    out_prod = out_prod * (inn_prod * local_PD[i] + (self.oneMinusPD(i)))
            res = res + sign * out_prod
            if showStatus and iteration%round((2**len(present_findings))/8)==0:
                print(iteration/(2**len(present_findings))*100,'%')
            iteration += 1
        return res
    
    def PFindings_not_di(self,present_findings, absent_findings, d_i = None):
        '''    
        Implements equation 11 from paper: The probability that a mixture of findings will be present or absent.
        
        If d_i is set the conditional probability of the findings given disease d_i=0 (i.e. not present) will be returned
        '''
        res = 0
        for F in _powerset(present_findings):
            sign = (-1)**len(F)
            out_prod = 1
            for i in range(self.N_disease()):
                inn_prod = 1
                for f in F+absent_findings:
                        inn_prod = inn_prod*(1-self.Q[f,i])

                if(i==d_i):
                    out_prod = out_prod # * inn_prod # This is the only change compared to other method.
                else:
                    out_prod = out_prod * ( inn_prod*self.PD[i]+(self.oneMinusPD(i)))
            res = res + sign*out_prod
        return res
    

    def posterior_d(self,present_findings, absent_findings, disease):
        '''    
        The posterior probability of disease given the findings
        '''
        num = self.PFindings(present_findings, absent_findings,disease)* self.PD[disease]
        den = self.PFindings(present_findings, absent_findings)
        print(den)
        if num==0 or den ==0:
            return 0
        else:
            return num/den
    
    def mple_dis_post_slow(self,diseases,present_findings, absent_findings):
        '''
        Gives posterior over each disease in diseases given the set of symptomps. Implemented in a brute force manner
        '''
        res = {}
        for i in diseases:
            res[i] = self.posterior_d(present_findings,absent_findings,i)
        return res
    
    
    # Caching strategies
    
    
    def mple_dis_post_fast(self,diseases,present_findings, absent_findings):
        '''
        Gives posterior over each disease in diseases given the set of symptomps. Implemented with a preprocessing step that caches
        some probabilities beforehand.
        '''
        res = {}
        # Preprocessing step
        P_only_di = {}
        for F in _powerset_tuple(tuple(present_findings)):
            for i in range(self.N_disease()):
                entry_F_i = 1
                for f in F+tuple(absent_findings):
                    entry_F_i = entry_F_i * (1-self.Q[f,i])
                P_only_di[F,i] = entry_F_i

        # Calculate denominator(joint probability)

        den = self.PFindings(list(present_findings),list(absent_findings))
        
        #Check if denominator is 0
        if den==0:
            for i in diseases:
                res[i] = 0
        else:
            # Calculate posterior for each query disease
            for i in diseases:
                res_sum = 0
                for F in _powerset_tuple(tuple(present_findings)):
                    sign = (-1)**len(F)
                    out_prod = 1
                    for ii in range(self.N_disease()):
                        if ii != i:
                            out_prod = out_prod*(P_only_di[F,ii]*self.PD[ii]+(self.oneMinusPD(i)))
                        else:
                            out_prod = out_prod*(P_only_di[F,ii])
                    res_sum = res_sum + sign*out_prod

                res[i] = res_sum*self.PD[i]/den



        return res

    def mple_dis_post_fast_v3(self, diseases, present_findings, absent_findings, return_finding_prob=False):
        '''
        Gives posterior over each disease in diseases given the set of symptomps. Implemented with a preprocessing step that caches
        some probabilities beforehand.

        This revision only iterates over relevant parents in the making of the dicitonary

        '''
        res = {}
        relevant_parents = util.parentsOfFindings(self.Q,present_findings+absent_findings)
        # Preprocessing step
        P_only_di = {}
        dict2 = {}
        denn = 0
        for F in _powerset_tuple(tuple(present_findings)):
            F_entry = 1
            for i in relevant_parents:
                entry_F_i = 1
                for f in F + tuple(absent_findings):
                    entry_F_i = entry_F_i * (1 - self.Q[f, i])
                P_only_di[F, i] = entry_F_i
                # Extra preprocessing step: For each element in the powerset, calculate the associated product. This will be saved in a dict with an entry for each element.
                F_entry = F_entry * (entry_F_i * self.PD[i] + (1 - self.PD[i]))
            # Calculate the denominator here. This is the sum (with correct sign) of the entries in dict2
            denn = denn + (-1) ** len(F) * F_entry
            dict2[F] = F_entry

        # Calculate denominator(joint probability)
        # Check if denominator is 0
        if denn == 0:
            print('Probability of findings is 0. Division by zero.')
            for i in diseases:
                res[i] = 0


        else:
            # Calculate posterior for each query disease
            for i in diseases:
                if i in relevant_parents:
                    res_sum = 0
                    for F in _powerset_tuple(tuple(present_findings)):
                        sign = (-1) ** len(F)
                        P_only_di[F, i]
                        # for each entry in dict2 divide out the factor that is superfluous and multiply with the correct factor.
                        e = (dict2[F] / (P_only_di[F, i] * self.PD[i] + (1 - self.PD[i]))) * P_only_di[F, i]
                        res_sum = res_sum + sign * e
                    res[i] = res_sum * self.PD[i] / denn
                else:
                    res[i] = self.PD[i]

        if return_finding_prob == False:
            return res
        else:
            return res, denn
    
    
    
    def PPositiveFindings_sequential(self,result_old,powerset_old,find_new):
        '''Function that calculates probability of positive findings by adding them sequentially'''
        added_term=0
        
        new_sets = [s+[find_new] for s in powerset_old]
        res=0
        # Calculate contributions from new terms
        for F in new_sets:
            sign = (-1)**len(F)
            out_prod = 1
            for i in range(self.N_disease()):
                inn_prod = 1
                for f in F:
                    inn_prod = inn_prod*(1-self.Q[f,i])
                out_prod = out_prod * ( inn_prod*self.PD[i]+(self.oneMinusPD(i)))
            res = res + sign*out_prod
        result_new = result_old + res
        
        powerset_new = powerset_old+new_sets
        
        return result_new,powerset_new

                
    def calculate_innerprod_dic(self, positive_findings,negative_findings):
        '''Calculate the inner product dictionary '''
        P_only_di = {}
       
        for F in _powerset_tuple(tuple(positive_findings)):
            for i in range(self.N_disease()):
                entry_F_i = 1
                for f in F+tuple(negative_findings):
                    entry_F_i = entry_F_i * (1-self.Q[f,i])
                P_only_di[F,i] = entry_F_i
        powerset_generator = _powerset(positive_findings)
        return P_only_di, list(powerset_generator)
    
    def extend_ipdic_nfinding(self, dic, negative_finding):
        '''Function for updating the inner product dictionary with the information from a single negative finding'''
        for (a,b),c in dic.items():
            dic[a,b] = c*(1-self.Q[negative_finding,b])
        return dic
    
    def extend_ipdic_nfinding_ver2(self, dic, powerset_old, new_negative_finding, old_result):
        '''Function to updating the inner product dictionary with the information from a single negative finding'''
        res = 0
        for F in powerset_old:
            sign = (-1)**len(F)
            prod = 1
            for i in range(self.N_disease()):
                #Update dictionary entry:
                dic[tuple(F),i] = dic[tuple(F),i] * (1-self.Q[new_negative_finding,i])
                prod = prod*( dic[tuple(F),i] * self.PD[i] + (self.oneMinusPD(i)))
            res = res + sign*prod
        return dic, res

    
    def extend_ipdic_pfinding(self, dic, powerset_old, negative_findings,new_positive_finding, old_result):
        '''Function to update the dictionary with the new sets coming from the new finding and updating the probability result at the same time.'''
        #old_powerset_as_list = list(powerset_old)
        #new_sets = [s+[new_positive_finding] for s in old_powerset_as_list]
        new_sets = [s+[new_positive_finding] for s in powerset_old]
        added_res = 0
        for F in new_sets:
            sign = (-1)**len(F)
            prod=1
            for i in range(self.N_disease()):
                entry_F_i = 1
                for f in F+negative_findings:
                    entry_F_i = entry_F_i * (1-self.Q[f,i])
                dic[tuple(F),i] = entry_F_i
                prod = prod*(entry_F_i * self.PD[i] + (self.oneMinusPD(i)))
            added_res = added_res + sign*prod
        new_powerset = list(powerset_old) + new_sets
        new_result = old_result + added_res
        return dic, new_powerset, new_result
    
    def PFindings_dbased(self,positive_findings,negative_findings, dic, powerset):
        '''Function to calculate joint probability of the findings given that the inner product dictionary dic is already calculated'''
        res = 0
        #for F in _powerset_tuple(tuple(positive_findings)):
        for F in powerset:
            sign = (-1)**len(F)
            prod = 1
            for i in range(self.N_disease()):
                #print(dic[F,i])
                #assert type(dic[F,i]) == np.float64, 'Not float'
                F = tuple(F)
                prod = prod*( dic[F,i] * self.PD[i] + (self.oneMinusPD(i)))
            res = res + sign*prod
        return res    

    def add_finding(self, state, finding,dic = None, result = None,powerset = None, positive_findings = [], negative_findings = []):
        
        # Handle initial setup when dic and result is None
        #_________________________________________________
                    
        if dic is None:
            if state == 'positive':
                dic,powerset = self.calculate_innerprod_dic([finding],[])
                positive_findings = positive_findings + [finding]
            elif state =='negative':
                dic,powerset = self.calculate_innerprod_dic([],[finding])
                negative_findings = negative_findings + [finding]
            else:
                assert False, "State has to be negative or positive"
                
                
        if result is None:
            pass
            result = self.PFindings_dbased([finding],[],dic, powerset) if state == 'positive' else self.PFindings_dbased([],[finding],dic, powerset)
        #_________________________________________________
        
        outer = self
        
        class resultClass:
            def res(self):
                if result is not None:
                    return result
                else:
                    print('Error: no result yet')
            
            def add_finding(self,state,_finding):
                if state == 'positive':
                    new_dic,new_powerset,new_result = outer.extend_ipdic_pfinding(dic,powerset, negative_findings,_finding, result)
                    new_positive_findings = positive_findings + [_finding]
                    return outer.add_finding(_,_,dic = new_dic, result = new_result,powerset = new_powerset, positive_findings = new_positive_findings, negative_findings = negative_findings)
                if state == 'negative':
                    new_dic,new_result = outer.extend_ipdic_nfinding_ver2(dic, powerset, _finding, result)
                    new_negative_findings = negative_findings + [_finding]
                    return outer.add_finding(_,_,dic = new_dic, result = new_result, powerset = powerset, positive_findings = positive_findings, negative_findings = new_negative_findings)
                else:
                    assert False, "State has to be negative of positive"
        res = resultClass()
        return res
