import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import itertools as itr

def simulate_voter_model_dynamics(graph,steps, boundary_condition, bias, use_bc = False):
	"""
	Params:
	----------------------
	graph: networkx graph
	steps: int
		num of steps to run dynamics; starts with step 0 the initial measure ... T-1.
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
		for leaf,vote in zip(leaves,list(boundary_condition)):
			initial_state[leaf] = vote

	nx.set_node_attributes(graph,initial_state,'vote')
	

	history.append(initial_state)

	for step in range(steps-1):
		#exclude the leaves if use_bc is True
		if use_bc is True:
			nodes = non_leaves
		else:
			nodes = graph.nodes

		for node in nodes:

			neighbor_spins = [graph.node[nbr]['vote'] for nbr in graph.neighbors(node)] 
			prob = probability_of_one(sum(neighbor_spins))

			if np.random.rand() < prob:
				graph.node[node]['vote'] *= -1


		history.append(nx.get_node_attributes(graph,'vote'))

	return (history,graph)

def probability_of_one(sum_nbrs):
	if sum_nbrs > 0:
		return 1
	elif sum_nbrs == 0:
		return 0.5
	else:
		return 0

def simulate_mean_field_dynamics(kappa,p_0,T):
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
		old_prob = probs[-1]

		if old_prob > 0.5:
			new_prob = 1
		elif old_prob == 0.5:
			new_prob = 0.5
		else:
			new_prob = 0

		probs.append(new_prob)

	return probs

