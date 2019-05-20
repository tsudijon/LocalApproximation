import os 
import sys
sys.path.append("../SimulateProcesses")
import Dyadic_measures as measures

import Contact.JittedContactLocalApprox as local 

# import standard libraries
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

import ast
import importlib

importlib.reload(local)


### applying jit code
q = 0.35 # recovery rate
p = 1 - q #infection rate
kappa = 4
initial_history = 2
initial_measure = measures.get_random_dyadic_measure(k = initial_history,dimension = kappa + 1)

#local.simulate_local_approximation(1, initial_measure, p, q)
(history,measure) = local.simulate_k_approximation(100, initial_measure, 1, p, q)
