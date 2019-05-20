import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import itertools as itr

from numba import prange, jit, njit


################################################
################################################
################################################
### Simulate Contact Process dynamics ###

def simulate_Contact_Process_dynamics(graph, steps, p,q, bd_condition, bias, mc_samples, use_bc = False):
	'''
	p = infection rate
	q = recovery rate
	'''

	if use_bc:
		nodes = non_leaves
	else:
		nodes = list(graph.nodes)

	leaves = [k for k,v in dict(graph.degree).items() if v == 1] ##
	non_leaves = list(set(graph.nodes) - set(leaves)) ##

	neighbors = {node:list(graph.neighbors(node)) for node in nodes} 
	neighbors_array = np.zeros(( len(nodes), graph.degree[0] ) )

	for key, neighbors in neighbors.items():
		neighbors_array[key,:] = np.array(neighbors,dtype = np.int32)

	biased_p = (1+bias)/2

	return run_dynamics(biased_p, len(nodes), neighbors_array, steps, p, q, mc_samples)

@njit(cache = True)
def run_dynamics(biased_p, nodes, neighbors, steps, p, q, mc_samples):

	averaged_history = np.zeros((4,steps))
	for i in prange(mc_samples):
		print(i)

		initial_state = [1 if p < biased_p else 0 for p in np.random.rand(nodes)]

		history = np.zeros((nodes,steps),dtype = np.int64)
		history[:,0] = initial_state

		for step in prange(1,steps - 1):
			# this is an asynchronous update.
			old_state = history[:,step - 1].copy()
			new_state = history[:,step - 1].copy()

			for node in range(nodes):
				neighbors_infected = np.array([old_state[int(nbr)] for nbr in neighbors[node,:]])  ## change this to update the array directly
				current_condition = old_state[node]
				prob = contact_flip_probability(current_condition,p,q, neighbors_infected)

				# flip the infected condition
				if np.random.rand() < prob:
					new_state[node] = 1 - new_state[node]

			history[:,step] = new_state

		root_nbr_dynamics = history[np.array([0,1]),:]

		for i in range(steps):
			averaged_history[2*root_nbr_dynamics[0,i] + root_nbr_dynamics[1,i], i] += 1

	return averaged_history/mc_samples

@njit(cache = True)
def contact_flip_probability(current_condition, p,q,neighbors_infected):
	if current_condition == 1:
		return q

	else:
		averaged_neighbors = np.sum(neighbors_infected)/len(neighbors_infected)
		return p*averaged_neighbors

@njit(cache = True)
def contact_one_probability(current_condition, p,q, neighbors_infected):
	'''
	returns the probability of a node transitioning to one at the next time step
	'''
	if current_condition == 1:
		flip_prob = contact_flip_probability(current_condition,p,q,neighbors_infected)
		return 1 - flip_prob

	else:
		flip_prob = contact_flip_probability(current_condition,p,q,neighbors_infected)
		return flip_prob

def simulate_mean_field_dynamics(kappa,p_0,T,p,q):
	"""
	Computes the probabilities of a node being 1 for the contact process, using the mean field
	Assumption
	"""
	probs = [p_0]
	for i in range(T):
		p_old = probs[-1]
		p_new = (1-q)*p_old + p*p_old*(1-p_old)
		probs.append(p_new)
	return probs



