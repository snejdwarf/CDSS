from src.quickscore import Quickscore
from src.variationalInference import variationalinference
import time
import random
import numpy as np

random.seed(42)

#Create quickscore object and model containing nd diseases and nf symptoms
nd = 400
nf = 800
qs = Quickscore.Quickscore()

qs.generateRandomTables(nd,nf)
Q = qs.Q
PD = qs.PD

#Define a set of symptoms that are present in a patient
positive_findings = [34,56,78,79,82,85,87,90,91,95,100,105,108]

#Calculate the exact probability of having the symptoms
tic = time.time()
print('Calculating exact probability...')
res=qs.probability_of_findings(positive_findings, []);
tac = time.time()
print("Exact probability",res, "Time: ",tac-tic);

#Variational inference: Calculate an upper and a lower bound on the exact probability
tic = time.time()
vi = variationalinference.VI()
res_vi=vi.joint_probability_bound(Q,PD,positive_findings,5)
tac = time.time()
print(res_vi, tac-tic)