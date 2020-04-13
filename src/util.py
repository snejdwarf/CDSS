import operator
import numpy as np
from functools import reduce


def almost_equal(a, b, _compare_precision=1e-13):
    return abs(a - b) <= _compare_precision


def compare_results(res_a, res_b, _compare_precision=1e-13):
    for a, b in zip(res_a, res_b):
        if not almost_equal(a, b, _compare_precision=_compare_precision):
            print("\n%.20lf\n%.20lf\n\t%.20lf\t%.20lf" % (a, b, a-b, _compare_precision))
            return False

    return True


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def powerset(set, prev_sets=None, indices=None):
    if prev_sets is None:
        prev_sets = [()]
    if indices is None:
        indices = [0]
    cur_sets = []
    cur_indices = []

    for index, prev in zip(indices, prev_sets):
        # print(index, prev)
        yield prev  # We yield prev, so that we also get the empty set
        for i in range(index, len(set)):
            new_set = prev + (set[i],)
            cur_sets += [new_set]
            cur_indices += [i + 1]
            # yield new_set  # See yield prev

    if len(cur_sets) > 0:
        yield powerset(set, cur_sets, cur_indices)


def __powerset(s):
    if len(s) == 0:
        yield s
    else:
        for set in powerset(s[1:]):
            yield [s[0]] + set
            yield set
     
    
def findingRelatedDis(Q, finding):
    '''
    Given a finding return indexes of all connected diseases (parents of the finding)
    '''
    res=np.argwhere(Q[finding, :] > 0)
    res = np.array([e[0] for e in res])
    return res

def diseaseRelatedFindings(Q, disease):
    '''
    Given a disease return indexes of all connected findings (children of the disease)
    '''
    res=np.argwhere(Q[:,disease] > 0)
    res = np.array([e[0] for e in res])
    return res

def nonZeroFindings(Q):
    '''
    Return all the findings that actually has a probability of occuring (are connected with some disease)
    '''
    NS = Q.shape[0]
    res = []
    for i in range(NS):
        if len(np.nonzero(Q[i,:])[0])>0:
            res.append(i)
    return res


def parentsOfFindings(Q, findings):
    if type(findings)==tuple:
        findings = list(findings)
    assert type(findings) == list
    '''Returns the set of parents to all of the findings'''
    res = np.array([])
    for finding in findings:
        res= np.union1d(findingRelatedDis(Q, finding),res,)
    return list(res.astype(np.int))

def rowNormalize2DArray(A):
    res = np.array(A)
    A_rowsums = A.sum(1)
    res = res / A_rowsums[:, np.newaxis]
    return res

def colNormalize2DArray(A):
    res = np.array(A)
    A_colsums = A.sum(0)
    res = res / A_colsums[np.newaxis, :]
    return res

def normalize2DArray(A):
    R = np.array(A) / np.sum(A)
    return R
