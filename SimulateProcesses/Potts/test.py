import pandas as pd 
import itertools as itr
import functools
import numpy as np 

import ast
import importlib

import Dyadic_measures as measures

import Potts.Potts_Model as potts 
import Potts.JittedPottsLocalApprox as local




### applying jit code
kappa = 2
initial_history = 2
p_0 = [0.1,0.6,0.3]
num_states = 3
beta = 0.1

#initial_measure = measures.get_random_dyadic_measure(k = initial_history,dimension = kappa + 1)

initial = np.zeros(tuple([num_states]*(kappa+1)))
for idx in itr.product(list(range(num_states)),repeat = (kappa+1)):
	idx_counts = [sum(np.array(idx) == i) for i in range(num_states)]
	initial[idx] = np.prod(np.power(p_0,idx_counts))

initial_measure = measures.nary_measure(initial,k = 1, dimension = kappa + 1, n = num_states)

#local.simulate_local_approximation(1, initial_measure, p, q)
importlib.reload(local)
(measure_history,measure) = local.simulate_k_approximation(2, initial_measure, 1, beta = beta, num_states = num_states)



### Applying dynamics 
steps = 1000
depth = 10
boundary_condition = [1]*(kappa**depth)
g = graphs.create_regular_tree(kappa,depth)
use_bc = False

(history,graph) = voter.simulate_voter_model_dynamics(g,steps, boundary_condition, bias, use_bc)

print(len(history[-1].values()))
print(sum(history[-1].values()))