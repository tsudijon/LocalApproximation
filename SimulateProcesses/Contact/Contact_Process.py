import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import itertools as itr


################################################
################################################
################################################
### Simulate Contact Process dynamics ###

def simulate_Contact_Process_dynamics(graph, steps,p,q, bd_condition, bias, use_bc = False):
	'''
	p = infection rate
	q = recovery rate
	'''
	history = []

	leaves = [k for k,v in dict(graph.degree).items() if v == 1]
	non_leaves = list(set(graph.nodes) - set(leaves))

	biased_p = (1+bias)/2

	initial_state = [1 if p < biased_p else 0 for p in np.random.rand(len(graph.nodes))]

	initial_state = {idx:state for idx,state in enumerate(initial_state)} # convert the spins into a dict

	if use_bc:
		# update the leaves of the graph 
		for leaf,infected in zip(leaves,list(bd_condition)):
			initial_state[leaf] = infected

	nx.set_node_attributes(graph,initial_state, 'infected')
	history.append(initial_state)

	for step in range(steps):
		if use_bc:
			nodes = non_leaves
		else:
			nodes = graph.nodes

		# this is an asynchronous update.
		old_state = history[-1].copy()
		new_state = history[-1].copy()

		for node in nodes:
			neighbors_infected = [old_state[nbr] for nbr in graph.neighbors(node)] 
			current_condition = old_state[node]
			prob = contact_one_probability(current_condition,p,q, neighbors_infected)

			# flip the infected condition
			if np.random.rand() < prob:
				new_state[node] = 1
			else:
				new_state[node] = 0

		history.append(new_state)

	return (history,graph)


def contact_flip_probability(current_condition, p,q,neighbors_infected):
	if current_condition == 1:
		return q

	else:
		averaged_neighbors = sum(neighbors_infected)/len(neighbors_infected)
		return p*averaged_neighbors

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



