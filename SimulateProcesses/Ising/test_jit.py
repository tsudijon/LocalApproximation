import os 
import sys
sys.path.append("../")
import Dyadic_measures as measures

import JittedIsingLocalApprox as local 

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
beta = 0.01
p = 0.9
kappa = 2
initial_history = 1
initial_measure = measures.get_random_dyadic_measure(k = initial_history,dimension = kappa + 1)

#local.simulate_local_approximation(1, initial_measure, p, beta)
local.simulate_k_approximation(100, initial_measure, 1, p, beta)



##



