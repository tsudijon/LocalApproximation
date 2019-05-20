import os 
import sys


# import standard libraries
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

sys.path.append("../SimulateProcesses/")
import Dyadic_measures as dm

import ast
import importlib

from numba import jit, njit, prange

def simulate_local_approximation(steps, measure, p, q):
	'''
	Step forward with the transition operator
	'''
	#keep track of consecutive l2 differences

	consec_diff = []

	current_measure = measure
	dimension = len(current_measure.shape)
	side_length = len(current_measure)
	history_length = int(np.log2(side_length))
	for i in range(steps):
		new_measure = apply_nonlinear_markov_operator(True, dimension,current_measure, side_length,history_length, p, q)

		diff = np.sqrt( ((new_measure - dm.embed(current_measure, history_length +1))**2).sum())
		# measure difference here
		dimension = len(new_measure.shape)
		side_length = len(new_measure)
		history_length = int(np.log2(side_length))
		consec_diff.append(diff)

		current_measure = new_measure

	return (consec_diff,current_measure)

def simulate_k_approximation(steps, measure, k, p, q):
		'''
		Applies the smoothening operator to the simulate_local_approximation function
		'''
		consec_diff = []

		current_measure = measure
		dimension = len(current_measure.shape)
		side_length = len(current_measure)
		history_length = int(np.log2(side_length))

		for i in range(steps):
			new_measure = apply_nonlinear_markov_operator(True, dimension, current_measure, side_length, history_length, p, q)
			
			if history_length + 1 > k:
				smoothed_measure = dm.smoothen(new_measure,history_length)

				side_length = len(smoothed_measure)
				history_length = int(np.log2(side_length))

				#record difference
				consec_diff.append(np.sqrt(((smoothed_measure - current_measure)**2).sum()))

				# update current measure
				current_measure = smoothed_measure

			else:
				side_length = len(new_measure)
				history_length = int(np.log2(side_length))
				
				# record difference
				diff = np.sqrt( ((new_measure - dm.embed(current_measure, history_length))**2).sum())
				consec_diff.append(diff)

				# update current_measure
				current_measure = new_measure

		return (consec_diff,current_measure)


@jit(cache = True,parallel = True)
def apply_nonlinear_markov_operator(measure_diff,dimension, current_measure,side_length,history_length, p, q):
	'''
	Should be the transition operator (string together the functions below)
	defines a function to update the current measure of the graph

	Returns the measure diff if paramtert is passed in , else None
	'''
	#dimension = len(current_measure.shape)
	

	new_side_length = 2*side_length
	new_measure = np.zeros([new_side_length]*dimension, dtype = np.float64)


	## Compute conditional expectations here
	cond_exp = compute_conditional_expectation(side_length, current_measure, p, q, dimension - 1)

	# update new measure by integrating the kernel over old measure
	# can probably parallellize this to make it faster

	args = tuple([list(range(side_length))]*dimension)
	states = np.array(np.meshgrid(*args)).T.reshape(-1,dimension)

	for j in prange(len(states)):
		state = states[j]
		print(state)
		#get the dictionary of where to map to with probabilities

		# get the product of bernoullis
		bernoulli_ps = np.zeros(dimension)

		for i in range(dimension):

			if i == 0:

				current_sign = int(2*((tuple(state)[i]))/side_length)

				sum_neighbors = np.sum( (2*np.array(tuple(state)[1:])/side_length).astype(np.int32) )
				functional = contact_one_probability(current_sign, p, q, sum_neighbors, dimension - 1) 

			else:
				functional = cond_exp[tuple(state)[i],tuple(state)[0]]

			bernoulli_ps[i] = functional



		bernoulli_ps = np.array([1-bernoulli_ps,bernoulli_ps])

		# create new measure
		new_side_length = 2*side_length
		probability_kernel = np.zeros([new_side_length]*dimension)

		args = tuple([[0,1]]*dimension)
		for transition in np.array(np.meshgrid(*args)).T.reshape(-1,dimension):

			probs = bernoulli_ps[tuple(transition),range(dimension)]

			# in the coordinates of the new measure, which has double the side length
			new_state = np.array(state) + side_length*np.array(transition) 

			probability_kernel[tuple(new_state)] = np.prod(probs)


		prob = current_measure[tuple(state)]

		# update the new measure 
		new_measure = new_measure + probability_kernel*prob

	# measure the difference between the old and new measures

	return new_measure

@njit(cache=True)
def compute_conditional_expectation(side_length, current_measure, p, q,kappa):
	'''
	should only be used where node  = 1,...,d
	state is a tuple of length kappa+1 where kappa is the degree of the tree
	'''

	## initialize container
	conditional_expectations = np.zeros(shape = (side_length,side_length))

	for i in range(side_length):
		for j in range(side_length):
		
			
			current_sign = int(2*i/side_length) 

			### get the conditioned measure ###

			conditioned_measure = current_measure[i,j,:] #plug in the first few coordinates

			s = conditioned_measure.sum()
			if s != 0:
				conditioned_measure_sum = s
			else:
				conditioned_measure_sum = 1
				conditional_expectations[i,j] = 0
				continue

			# test for whether index is in conditional measure
			in_conditional_measure = lambda idx: (idx[0] == i) and (idx[1] == j)

			### calculate the  conditional expectation ###
			contact_functional = lambda idx: contact_one_probability(current_sign,
								 p, q, sum(fcd(np.array(idx[1:]))) )

			# compute the expectation by looping through indices of the conditional measure
			weighted_values = [prob*contact_one_probability(current_sign,
								 p, q, np.sum( (2*np.array(idx[1:])/side_length).astype(np.int32) ), kappa) for idx, prob in \
								np.ndenumerate(current_measure) if in_conditional_measure(idx)]	


			conditional_expectations[i,j] = np.sum(np.array(weighted_values,dtype = np.float64))/conditioned_measure_sum

	return conditional_expectations

@njit(cache=True)
def contact_flip_probability(current_condition, p,q,neighbors_infected, kappa):

	if current_condition == 1:
		return q
	else:
		a = np.array(neighbors_infected,dtype = np.int32)
		averaged_neighbors = np.sum(a)/kappa
		return p*averaged_neighbors

@njit(cache=True)
def contact_one_probability(current_condition, p,q, neighbors_infected, kappa):
	'''
	returns the probability of a node transitioning to one at the next time step
	'''
	if current_condition == 1:
		flip_prob = contact_flip_probability(current_condition,p,q,neighbors_infected, kappa)
		return 1 - flip_prob

	else:
		flip_prob = contact_flip_probability(current_condition,p,q,neighbors_infected, kappa)
		return flip_prob