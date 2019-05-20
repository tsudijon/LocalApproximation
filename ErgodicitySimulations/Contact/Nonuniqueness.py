import os 
import sys
sys.path.append("../../SimulateProcesses/")
sys.path.append("../../SimulateProcesses/Contact")
import Create_Graphs as graphs 
import Dyadic_measures as measures
import JittedContactProcess as Contact 
import JittedContactLocalApprox as local
import ContactMonitor as monitor
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

kappa = 3
initial_history = 1
bias = 0
biased_p = (1+bias)/2
p = 0.9
q = 0.1
#initial_measure = measures.get_random_dyadic_measure(k = initial_history,dimension = kappa + 1)

initial = np.zeros(tuple([2]*(kappa+1)))
for idx in itr.product([0,1],repeat = (kappa+1)):
	initial[idx] = ((biased_p)**(sum(idx)))*((1-biased_p)**(kappa + 1 - sum(idx)))

initial_measure = measures.dyadic_measure(initial,k = 1, dimension = kappa + 1)

#local.simulate_local_approximation(1, initial_measure, p, q)
(measure_history,measure) = local.simulate_k_approximation(1000, initial_measure, initial_history,p,q)

zero_measure = np.zeros_like(initial_measure)
zero_measure[tuple([0]*(kappa+1))] = 1

(measure_history,measure) = local.simulate_k_approximation(10, zero_measure, initial_history,p,q)

