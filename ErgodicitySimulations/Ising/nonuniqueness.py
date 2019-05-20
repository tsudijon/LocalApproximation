import os 
import sys
sys.path.append("../../SimulateProcesses/")
sys.path.append("../../SimulateProcesses/Ising")
import Create_Graphs as graphs 
import Dyadic_measures as measures
import Ising_Model as Ising 
import JittedIsingLocalApprox as local
import IsingMonitor as monitor
import Misc

# import standard libraries
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

import ast
import importlib
import argparse
import networkx as nx

# Exhibit nonuniqueness

kappa = 4
initial_history = 1
p = 0.9
#initial_measure = measures.get_random_dyadic_measure(k = initial_history,dimension = kappa + 1)

def create_initial_measure(biased_p):
	initial = np.zeros(tuple([2]*(kappa+1)))
	for idx in itr.product([0,1],repeat = (kappa+1)):
		initial[idx] = ((biased_p)**(sum(idx)))*((1-biased_p)**(kappa + 1 - sum(idx)))

	initial_measure = measures.dyadic_measure(initial,k = 1, dimension = kappa + 1)

	return initial_measure

betas = np.linspace(0,1,20)
biased_ps = np.linspace(0,1,10)
measure_norms = np.zeros((len(betas),len(biased_ps)))
node_probs = np.zeros((len(betas),len(biased_ps)))
for i,beta in enumerate(betas):
	for j,biased_p in enumerate(biased_ps):
			initial_measure = create_initial_measure(biased_p)
			(measure_history,measure) = local.simulate_k_approximation(500, initial_measure, 1, p, beta)
			measure_norms[i,j] = np.sum(np.power(measure,2))
			node_probs[i,j] = measure[tuple([0]*(kappa + 1))]

plt.figure()
ax = plt.axes()
for i in range(len(biased_ps)):
	noise = np.random.normal(0,0,len(betas))
	ax.scatter(betas,node_probs[:,i] + noise,s = 10)

plt.title("1-approximation stationary measures, Ising Model, kappa = {}".format(kappa))
plt.xlim(-0.1,1.1)
plt.xlabel("Inverse Temperature Beta")
plt.ylabel("Probability of Zero State")
plt.axvline(x=0.549306,linestyle='--')
plt.show()
