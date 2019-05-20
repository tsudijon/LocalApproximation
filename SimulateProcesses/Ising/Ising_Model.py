import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import itertools as itr

import JittedIsingLocalApprox as Local
import Create_Graphs

import importlib
importlib.reload(Local)

from numba import jitclass

# ways to improve this: can make IsingKApproximation a subclass of both Ising Local Approx and the KApproximation


################################################
################################################
################################################
### Simulate Glauber dynamics ###

def simulate_parallel_glauber_dynamics(graph,steps,p,beta, boundary_condition, bias, use_bc = False):
	"""
	Params:
	----------------------
	graph: networkx graph
	steps: int
		num of steps to run dynamics; starts with step 0 the initial measure ... T-1.
	p: float in [0,1]
		asynchronous update parameter
	beta: positive float
		inverse temperature param for the Ising model
	boundary_condition: list of length kappa^depth - each element is +,- 1
		boundary condition at which to fix for the Ising model
	bias: float between -1,1
		initial probability is then given by (1 + bias)/2, Bernoulli
	"""
	history = []
	leaves = [k for k,v in dict(graph.degree).items() if v == 1]
	non_leaves = list(set(graph.nodes) - set(leaves))

	biased_p = (1+bias)/2

	initial_state = [1 if p < biased_p else -1 for p in np.random.rand(len(graph.nodes))]
	initial_state = {idx:state for idx,state in enumerate(initial_state)} # convert the spins into a dict

	if use_bc is True:
	
		# update the leaves of the graph 
		for leaf,spin in zip(leaves,list(boundary_condition)):
			initial_state[leaf] = spin

	nx.set_node_attributes(graph,initial_state,'spin')
	history.append(initial_state)

	for step in range(steps):
		#exclude the leaves if use_bc is True
		if use_bc is True:
			nodes = non_leaves
		else:
			nodes = graph.nodes

		old_state = history[-1].copy()
		for node in nodes:

			neighbor_spins = [old_state[nbr] for nbr in graph.neighbors(node)] 
			current_sign = old_state[node]
			prob = probability_of_flip(current_sign,p,beta, sum(neighbor_spins))

			if np.random.rand() < prob:
				graph.node[node]['spin'] *= -1


		history.append(nx.get_node_attributes(graph,'spin'))

	return (history,graph)


def visualize_Glauber_dynamics(graph,history):
	
	pos = nx.spring_layout(graph)

	for node_values in history[::-1]:
		plt.figure()
		nx.draw(graph, node_color = list(node_values.values()), node_size = 20, pos = pos)
	
	plt.show()

def simulate_mean_field_dynamics(kappa,p_0,T,p,beta):
	"""
	kappa: int
		degree of the tree
	p_0: float [0,1]
		probability of initial value being one.
	T: int
		time at which to stop
	p: float [0,1]
		param for parallel Ising
	beta: float > 0
		inverse temperature
	Output:
		a list of probabilities of a node being 1.
	"""
	probs = [p_0]
	for i in range(T-1):
		prev_p = probs[-1]
		p_given_minus_one = probability_of_flip(-1,p,beta,kappa*(2*prev_p - 1))
		p_given_one = (1 - probability_of_flip(1,p,beta,kappa*(2*prev_p - 1)))
		next_p = p_given_one*(prev_p) + p_given_minus_one*(1-prev_p)

		probs.append(next_p)

	return probs

def probability_of_flip(current_sign,p,beta, sum_neighbors):
	'''
	compute probability in Ising model - ratio of exponentials. This is for the parallel Glauber dynamics
	'''
	exp_term = np.exp(-beta*current_sign*sum_neighbors)
	inv_exp_term = np.exp(beta*current_sign*sum_neighbors)
	p_given_chosen = exp_term/(exp_term + inv_exp_term)

	return p*p_given_chosen

def probability_of_one(current_sign, p, beta, sum_neighbors):
	'''
	compute probability of transitioning to the one state
	'''
	exp_term = np.exp(-beta*sum_neighbors)
	inv_exp_term = np.exp(beta*sum_neighbors)

	if current_sign == 1:
		probability = 1 - p*exp_term/ (exp_term + inv_exp_term)
	else:
		probability = p * inv_exp_term/ (exp_term + inv_exp_term)

	return probability




