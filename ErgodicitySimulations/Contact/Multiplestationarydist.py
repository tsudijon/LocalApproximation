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

kappa = 2
initial_history = 1
q = 0.1

##########################################################################################
##########################################################################################
##########################################################################################
def create_initial_measure(biased_p):
	initial = np.zeros(tuple([2]*(kappa+1)))
	for idx in itr.product([0,1],repeat = (kappa+1)):
		initial[idx] = ((biased_p)**(sum(idx)))*((1-biased_p)**(kappa + 1 - sum(idx)))

	initial_measure = measures.dyadic_measure(initial,k = 1, dimension = kappa + 1)

	return initial_measure

 



