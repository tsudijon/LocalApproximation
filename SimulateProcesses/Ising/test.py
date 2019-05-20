import os 
import sys
sys.path.append("../")
import Create_Graphs as graphs 
import Dyadic_measures as measures
import Ising_Model as Ising 
import Monitor_Processes as monitor
import Misc

# import standard libraries
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

import ast
import importlib


#Ising Params
beta = 0.1
p = 0.9

kappa = 5
initial_history = 3
initial_measure = measures.get_random_dyadic_measure(k = initial_history,dimension = kappa + 1)
p = Ising.IsingLocalApproximation(kappa, initial_measure, initial_history, beta, p)

p.compute_conditional_expectation()

p.apply_nonlinear_markov_operator(True)





