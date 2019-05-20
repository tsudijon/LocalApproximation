import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import itertools as itr

import Local_Approximation
import Create_Graphs

################################################
################################################
################################################
### Simulate Contact Process dynamics ###

def simulate_Contact_Process_dynamics(graph, steps, bd_condition, bias, p, q):
	'''
	p = infection rate
	q = recovery rate
	'''
	history = []

	leaves = [k for k,v in dict(graph.degree).items() if v == 1]
	non_leaves = list(set(graph.nodes) - set(leaves))

	biased_p = (1+bias)/2

	initial_state = [1 if p < biased_p else -1 for p in np.random.rand(len(graph.nodes))]

	initial_state = {idx:state for idx,state in enumerate(initial_state)} # convert the spins into a dict

	# update the leaves of the graph 
	for leaf,infected in zip(leaves,list(boundary_condition)):
		initial_state[leaf] = infected

	nx.set_node_attributes(graph,initial_state, 'infected')
	

	for step in range(steps):
		for node in non_leaves:

			neighbors_infected = [nbr['infected'] for nbr in graph.neighbors(node)] 
			current_condition = graph.node[node]['infected']
			prob = probability_of_flip(current_condition,p,q, neighbors_infected)

			# flip the infected condition
			if np.random.rand() < prob:
				graph.node[node]['infected'] = 1 - graph.node[node]['infected']

		history.append(nx.get_node_attributes(graph,'infected'))

	return (history,graph)


def contact_flip_probability(current_condition, p,q,neighbors_infected):

	assert current_condition in [0,1], "current condition must be either 0,1"

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



def simulate_mean_field_dynamics(p_0,kappa,T,p,q):
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

################################################
################################################
################################################
### Define Contact Process-specific local approximation in nonlinear markov chain form ###

class ContactLocalApproximation(LocalApproximationSimulation):
	'''
	p is the infection rate,
	q is the recovery rate
	'''
	def __init__(self, graph,k, initial_measure = None, initial_history = 1, p = 0.5, q = 0.5):
		super(ContactLocalApproximation, self).__init__(graph,k, initial_measure, initial_history)
		self.p = p
		self.q = q 
		self.phi = lambda x: x

	def get_bernoulli_functional(self, node, state):
		'''
		get the probability of being 1 at the next time step for each node, given the state of the process.
		'''
		assert node <= self.dimension

		# return the ordinary probability
		if node == 0:

			fcd = lambda x: Local.get_fcd(x,self.side_length)
			current_sign = fcd(state[node]) 

			assert current_sign == 1 or current_sign == 0, "current sign should be 0,1"

			sum_neighbors = sum(fcd(np.array(state[1:])))
			functional = contact_one_probability(current_sign, self.p, self.q, sum_neighbors) 

		else:
			functional = self.compute_conditional_expectation(node,state)

		return functional

	def compute_conditional_expectation(self, node, state):
		'''
		should only be used where node  = 1,...,d
		state is a tuple of length kappa+1 where kappa is the degree of the tree
		'''
		assert node > 0, "node should not be the root" 

		fcd = lambda x: Local.get_fcd(x,self.side_length)
		current_sign = fcd(state[node]) 

		assert current_sign == 1 or current_sign == 0, "current sign should be 0,1" 

		### get the conditioned measure ###

		conditioned_measure = self.current_measure[state[node],state[0],:] #plug in the first few coordinates

		if conditioned_measure.sum() != 0:
			conditioned_measure_sum = conditioned_measure.sum()
		else:
			conditioned_measure_sum = 1

		# test for whether index is in conditional measure
		in_conditional_measure = lambda idx: (idx[0] == state[node]) and (idx[1] == state[0])

		### calculate the  conditional expectation ###
		contact_functional = lambda idx: contact_one_probability(current_sign,
							 self.p, self.q, sum(fcd(np.array(idx[1:]))) )

		# compute the expectation by looping through indices of the conditional measure
		weighted_values = [prob*contact_functional(idx) for idx, prob in \
							np.ndenumerate(self.current_measure) if in_conditional_measure(idx)]		
		expectation = sum(weighted_values)

		return expectation/conditioned_measure_sum

class ContactKApproximation(KApproximation):
	pass




