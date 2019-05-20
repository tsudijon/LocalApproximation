import os 
import sys
sys.path.append("../")
import Dyadic_measures as measures

import AutomataLocalApprox as local 
import Automata as oscillator

# import standard libraries
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

import ast
import importlib

importlib.reload(local)



if __name__ == "__main__":
	### applying jit code
	p = 0.1 # p in [0,0.5]
	k = 1
	kappa = 2
	p_0 = 0.4

	initial = np.zeros(tuple([2]*(kappa+1)))
	for idx in itr.product([0,1],repeat = (kappa+1)):
		initial[idx] = (p_0**(sum(idx)))*((1-p_0)**(kappa + 1 - sum(idx)))

	#local.simulate_local_approximation(1, initial_measure, p, beta)
	print("running local approx")
	(history, measure) = local.simulate_k_approximation(100, initial, k, p)
	print(history)
	print(measure)

	print("running mean field")
	history = oscillator.simulate_mean_field_dynamics(10,kappa,p,p_0)
	print(history)
