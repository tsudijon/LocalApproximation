import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import itertools as itr

import Create_Graphs

import importlib

from numba import jitclass

# ways to improve this: can make IsingKApproximation a subclass of both Ising Local Approx and the KApproximation


################################################
################################################
################################################
### Simulate Glauber dynamics ###

def simulate_parallel_glauber_dynamics(graph, num_states, steps,beta, boundary_condition, bias, use_bc = False):
	"""
	We're simulating synchronous dynamics where the nodes update at the same time
	Params:
	----------------------
	graph: networkx graph
	num_states: int
		number of states for the Potts Model
	steps: int
		num of steps to run dynamics; starts with step 0 the initial measure ... T-1.
	p: float in [0,1]
		asynchronous update parameter
	beta: positive float
		inverse temperature param for the Ising model
	boundary_condition: list of length kappa^depth - each element is +,- 1
		boundary condition at which to fix for the Ising model
	bias: list of positive floats, length num_states
		initial probability is then given by (bias)/sum(bias), Bernoulli
	"""
	history = []
	leaves = [k for k,v in dict(graph.degree).items() if v == 1]
	non_leaves = list(set(graph.nodes) - set(leaves))

	biased_p = bias/sum(bias)

	initial_state = vector_sample(biased_p, len(graph.nodes))
	initial_state = {idx:state for idx,state in enumerate(initial_state)} # convert the spins into a dict

	if use_bc is True:
	
		# update the leaves of the graph 
		for leaf,spin in zip(leaves,list(boundary_condition)):
			initial_state[leaf] = spin

	nx.set_node_attributes(graph,initial_state,'spin')
	

	history.append(initial_state)

	for step in range(steps-1):
		#exclude the leaves if use_bc is True
		if use_bc is True:
			nodes = non_leaves
		else:
			nodes = graph.nodes

		for node in nodes:

			neighbor_spins = [graph.node[nbr]['spin'] for nbr in graph.neighbors(node)] 

			desired_spin = np.random.choice(range(num_states))
			prob = potts_choice_probability(graph.node[node]['spin'], desired_spin, beta, neighbor_spins)

			if np.random.rand() < prob:
				graph.node[node]['spin'] = desired_spin


		history.append(nx.get_node_attributes(graph,'spin'))

	return (history,graph)

def vector_sample(ps, num_samples):
	bins = np.cumsum(ps)
	return list(np.digitize(np.random.rand(num_samples), bins, right = False))


def simulate_mean_field_dynamics(kappa,p_0,T,beta):
	"""
	kappa: int
		degree of the tree
	p_0: list of float [0,1]
		probabilities of initial values, must sum to one.
	T: int
		time at which to stop
	p: float [0,1]
		param for parallel Ising
	beta: float > 0
		inverse temperature
	Output:
		a list of probabilities of a node being 1.
	"""
	probs = [np.array(p_0)]
	states = len(p_0)

	for i in range(T-1):
		old_ps = np.array(probs[-1])
		new_ps = np.zeros(states)

		#update new ps
		for j in range(states):

			exp_old_ps = np.exp(-0.5*beta*kappa*old_ps)

			ps_given_prev = exp_old_ps[j]/(exp_old_ps[j] + exp_old_ps)
			p_transition = np.dot(ps_given_prev,old_ps)/states

			failed_transition_ps = exp_old_ps[j]/(exp_old_ps[j] + exp_old_ps) # same as the above
			p_fail_transition = old_ps[j]*sum(failed_transition_ps)/states

			new_ps[j] = p_transition + p_fail_transition

		probs.append(new_ps)

	return probs


def potts_choice_probability(current_sign, desired_sign,beta, neighbors):
	"""
	compute probability in Ising model - ratio of exponentials. This is for the parallel Glauber dynamics

	Params
	-------------
	neighbors: list of int
		neighbors between 0 .. q
	"""
	common_terms_desired = sum(np.array(neighbors) == desired_sign)
	common_terms_current = sum(np.array(neighbors) == current_sign)

	current = np.exp(-0.5*beta*common_terms_current)
	new = np.exp(-0.5*beta*common_terms_desired)
	prob = new/(current+new)


	return prob

