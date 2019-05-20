import os 
import sys


# import standard libraries
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

sys.path.append("../")
import Dyadic_measures as dm

import ast
import importlib

from numba import jit, njit, prange

def simulate_local_approximation(steps, measure, beta, num_states):
	'''
	Step forward with the transition operator
	'''
	#keep track of consecutive l2 differences

	consec_diff = []

	current_measure = measure
	dimension = len(current_measure.shape)
	side_length = len(current_measure)
	history_length = int(np.log(side_length)/np.log(num_states))
	for i in range(steps):
		new_measure = apply_nonlinear_markov_operator(True, dimension,current_measure, side_length,history_length, beta, num_states)

		diff = np.sqrt( ((new_measure - dm.nary_embed(current_measure, history_length +1, num_states))**2).sum())
		# measure difference here
		dimension = len(new_measure.shape)
		side_length = len(new_measure)
		history_length = int(np.log(side_length)/np.log(num_states))
		consec_diff.append(diff)

		current_measure = new_measure

	return (consec_diff,current_measure)

def simulate_k_approximation(steps, measure, k, beta, num_states):
	'''
	Applies the smoothening operator to the simulate_local_approximation function
	'''
	consec_diff = []

	current_measure = measure
	dimension = len(current_measure.shape)
	side_length = len(current_measure)
	history_length = int(np.log2(side_length))

	for i in range(steps):
		new_measure = apply_nonlinear_markov_operator(True, dimension, current_measure, side_length, history_length, beta, num_states)
		
		if history_length + 1 > k:
			smoothed_measure = dm.nary_smoothen(new_measure,history_length, num_states)

			side_length = len(smoothed_measure)
			history_length = int(np.log(side_length)/np.log(num_states))

			#record difference
			consec_diff.append(np.sqrt(((smoothed_measure - current_measure)**2).sum()))

			# update current measure
			current_measure = smoothed_measure

		else:
			side_length = len(new_measure)
			history_length = int(np.log(side_length)/np.log(num_states))
			
			# record difference
			diff = np.sqrt( ((new_measure - dm.nary_embed(current_measure, history_length, num_states))**2).sum())
			consec_diff.append(diff)

			# update current_measure
			current_measure = new_measure

	return (consec_diff,current_measure)


@jit(cache = True,parallel = True)
def apply_nonlinear_markov_operator(measure_diff, dimension, current_measure, side_length, history_length, beta, num_states):
	'''
	Should be the transition operator (string together the functions below)
	defines a function to update the current measure of the graph

	Returns the measure diff if paramtert is passed in , else None
	'''
	#dimension = len(current_measure.shape)
	

	new_side_length = num_states*side_length
	new_measure = np.zeros([new_side_length]*dimension, dtype = np.float64)


	## Compute conditional expectations here
	cond_exp = compute_conditional_expectation(side_length, current_measure, beta, num_states)

	# update new measure by integrating the kernel over old measure
	# can probably parallellize this to make it faster

	args = tuple([list(range(side_length))]*dimension)
	states = np.array(np.meshgrid(*args)).T.reshape(-1,dimension)

	for j in prange(len(states)):
		state = states[j]
		print(state)
		#get the dictionary of where to map to with probabilities

		# get the product of bernoullis
		multinoulli_ps = np.zeros((num_states, dimension))

		for i in range(dimension):

			if i == 0:

				current_sign = int(num_states*((tuple(state)[i]))/side_length)

				functional = np.zeros(num_states)

				for desired_sign in range(num_states):
					# compute the expectation by looping through indices of the conditional measure
					functional[desired_sign] = potts_total_probability(current_sign, desired_sign, beta, 
									(num_states*np.array(tuple(state)[1:])/side_length).astype(np.int32), num_states) 

			else:
				functional = cond_exp[tuple(state)[i],tuple(state)[0],:]

			multinoulli_ps[:,i] = functional

		# create new measure
		new_side_length = num_states*side_length
		probability_kernel = np.zeros([new_side_length]*dimension)

		args = tuple([list(range(num_states))]*dimension)
		for transition in np.array(np.meshgrid(*args)).T.reshape(-1,dimension):

			probs = multinoulli_ps[tuple(transition),range(dimension)]

			# in the coordinates of the new measure, which has double the side length
			new_state = np.array(state) + side_length*np.array(transition) 

			probability_kernel[tuple(new_state)] = np.prod(probs)


		prob = current_measure[tuple(state)]

		# update the new measure 
		new_measure = new_measure + probability_kernel*prob

	# measure the difference between the old and new measures

	return new_measure

@njit(cache=True)
def compute_conditional_expectation(side_length, current_measure, beta, num_states):
	'''
	should only be used where node  = 1,...,d
	state is a tuple of length kappa+1 where kappa is the degree of the tree
	'''

	## initialize container
	conditional_expectations = np.zeros(shape = (side_length,side_length,num_states))

	for i in range(side_length):
		for j in range(side_length):
		
			
			current_sign = int(num_states*i/side_length) 

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

			for desired_sign in range(num_states):
				# compute the expectation by looping through indices of the conditional measure
				weighted_values = [prob*potts_total_probability(current_sign, desired_sign, beta, 
								(num_states*np.array(idx[1:])/side_length).astype(np.int32), num_states) for idx, prob in \
									np.ndenumerate(current_measure) if in_conditional_measure(idx)]	

				conditional_expectations[i,j,desired_sign] = np.sum(np.array(weighted_values,dtype = np.float64))/conditioned_measure_sum

	return conditional_expectations

@njit(cache=True)
def potts_total_probability(current_sign, desired_sign, beta, neighbors, num_states):
	"""
	For the parallel Glauber dynamics, this function outputs the probability of a node at
	state current_sign transitioning to state desired_sign.
	Params
	-------------
	neighbors: list of int
		neighbors between 0 .. q
	"""


	if current_sign == desired_sign:
		prob_success = 1/2

		nbrs = np.array([np.sum(neighbors == i) for i in range(num_states)])
		exp_nbrs = np.exp(-0.5*beta*nbrs)

		prob_fail = np.sum(exp_nbrs[current_sign]/(exp_nbrs + exp_nbrs[current_sign]))

		return prob_success/num_states + prob_fail/num_states

	else:
		nbrs_current = np.sum(neighbors == current_sign)
		nbrs_desired = np.sum(neighbors == desired_sign)

		exp_desired = np.exp(-0.5*beta*nbrs_desired)
		exp_current = np.exp(-0.5*beta*nbrs_current)

		prob = exp_current/(exp_current + exp_desired)

		return prob/num_states



