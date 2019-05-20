import os 
import sys
sys.path.append("../../SimulateProcesses/")
import Dyadic_measures as measures
import Create_Graphs as graphs

import Voter.JittedVoterLocalApprox as local 
import Voter.Voter_Model as voter 

# import standard libraries
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

import ast
import importlib

importlib.reload(local)
importlib.reload(voter)

if __name__ == "__main__":
	### applying jit code
	kappa = 3
	initial_history = 1

	bias_vals = np.linspace(0,1,50)
	probs = np.array(bias_vals)

	for i in range(len(bias_vals)):
		print(i)
		bias = bias_vals[i]
		biased_p = (1+bias)/2
		#initial_measure = measures.get_random_dyadic_measure(k = initial_history,dimension = kappa + 1)

		initial = np.zeros(tuple([2]*(kappa+1)))
		for idx in itr.product([0,1],repeat = (kappa+1)):
			initial[idx] = ((biased_p)**(sum(idx)))*((1-biased_p)**(kappa + 1 - sum(idx)))

		initial_measure = measures.dyadic_measure(initial,k = 1, dimension = kappa + 1)

		#local.simulate_local_approximation(1, initial_measure, p, q)
		(measure_history,measure) = local.simulate_k_approximation(1000, initial_measure, initial_history)

		probs[i] = measure[tuple([1]*(kappa + 1))]

	print(probs)
