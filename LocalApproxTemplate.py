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

def simulate_local_approximation(steps, measure):
	'''
	Step forward with the transition operator, for a specified number of timesteps.
	Returns the ending measure and the succesive L2 differences between the measures.

	Params
	-------
	steps: int
		number of timesteps one wishes to run.
	measure: dyadic measure 
		input, starting measure. 
	'''

	consec_diff = []

	current_measure = measure
	dimension = len(current_measure.shape)
	side_length = len(current_measure)
	history_length = int(np.log2(side_length))

	# at each time step, step forward with the nonlinear markov operator and record
	for i in range(steps):
		new_measure = apply_nonlinear_markov_operator(True, dimension,current_measure, side_length,history_length)

		diff = np.sqrt( ((new_measure - dm.embed(current_measure, history_length +1))**2).sum())
		
		dimension = len(new_measure.shape)
		side_length = len(new_measure)
		history_length = int(np.log2(side_length))
		consec_diff.append(diff)

		current_measure = new_measure

	return (consec_diff,current_measure)

def simulate_k_approximation(steps, measure, k):
		'''
		Steps forward with the transition operator for the k approximation - applies
		nonlinear markov operator for the local recursion, then smoothens if necessary.

		Params:
		---------
		steps: int
			number of steps to run for
		measure: dyadic measure
			initial starting measure
		k: int
			number of past history steps to store.
		'''
		consec_diff = []

		current_measure = measure
		dimension = len(current_measure.shape)
		side_length = len(current_measure)
		history_length = int(np.log2(side_length))

		# at each time step, step forward with the nonlinear markov operator and record
		for i in range(steps):
			new_measure = apply_nonlinear_markov_operator(True, dimension, current_measure, side_length, history_length)
			
			if history_length + 1 > k:
				smoothed_measure = dm.smoothen(new_measure,history_length)

				side_length = len(smoothed_measure)
				history_length = int(np.log2(side_length))

				#record difference
				consec_diff.append(np.sqrt(((smoothed_measure - current_measure)**2).sum()))

				# update current measure
				current_measure = smoothed_measure

			# smoothen the measure if the history length is > k.
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
def apply_nonlinear_markov_operator(measure_diff,dimension, current_measure,side_length,history_length):
	'''
	Transition operator that pushes a measure forward by the 
	nonlinear Markov process defining the local approximation.
	This amounts to precomputing conditional expectation, 
	calculating bernoulli probabilities, and creating the updated
	measure.

	Returns the measure at the next time step.

	There might be some redundancy in the input parameters 
	(some can be deduced from others, but this is due to JIT/ Numba limitations).

	Params
	---------
	measure_diff: Boolean
		(Depr.) Boolean indicating whether to output L2 change in measures.
	dimension: int
		dimension of the input measure: num nodes in the local neighborhood
	current_measure: dyadic measure
		input measure
	side_length: int
		2^(number of steps kept track of so far), the side length of the input measure
	history_length: int
		number of steps kept track of so far
	'''
	

	new_side_length = 2*side_length
	new_measure = np.zeros([new_side_length]*dimension, dtype = np.float64)


	## Compute conditional expectations here
	cond_exp = compute_conditional_expectation(side_length, current_measure)

	# update new measure by integrating the kernel over old measure

	args = tuple([list(range(side_length))]*dimension)
	states = np.array(np.meshgrid(*args)).T.reshape(-1,dimension)

	# automatically parallelized by Numba
	for j in prange(len(states)):
		state = states[j]
		print(state)

		# compute bernoulli probabilities 
		bernoulli_ps = np.zeros(dimension)

		for i in range(dimension):
			if i == 0:
				sum_neighbors = np.sum( 2*((2*np.array(tuple(state)[1:])/side_length).astype(np.int32)) - 1 )

				### model specific computation ###
				functional =  probability_of_one(sum_neighbors)

			else:
				functional = cond_exp[tuple(state)[i],tuple(state)[0]]


			bernoulli_ps[i] = functional



		bernoulli_ps = np.array([1-bernoulli_ps,bernoulli_ps])

		# add to new measure the weighted probabilities of moving to the new 2^(dimension) possible states
		new_side_length = 2*side_length
		probability_kernel = np.zeros([new_side_length]*dimension)

		args = tuple([[0,1]]*dimension)
		for transition in np.array(np.meshgrid(*args)).T.reshape(-1,dimension):

			probs = bernoulli_ps[tuple(transition),range(dimension)]

			# in the coordinates of the new measure, which has double the side length
			new_state = np.array(state) + side_length*np.array(transition) 

			probability_kernel[tuple(new_state)] = np.prod(probs)


		prob = current_measure[tuple(state)]
		new_measure = new_measure + probability_kernel*prob


	return new_measure

@njit(cache=True)
def compute_conditional_expectation(side_length, current_measure):
	'''
	Precomputes the conditional expectations in the local recursion
	Output: 2D array of conditional expectations

	Params
	-----------
	side_length: int
		2^(number of steps kept track of so far), the side length of the input measure
	current_measure: dyadic measure
		measure with which the conditional expectation should be computed
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

			weighted_values = [prob*probability_of_one( \
					np.sum( 2*((2*np.array(idx[1:])/side_length).astype(np.int32)) - 1) ) for idx, prob in \
								np.ndenumerate(current_measure) if in_conditional_measure(idx)]	

			conditional_expectations[i,j] = np.sum(np.array(weighted_values,dtype = np.float64))/conditioned_measure_sum

	return conditional_expectations

@njit(cache=True)
def probability_of_one(sum_nbrs):
	"""
	Returns the Probability of a node flipping to the state 1, in a two state model.
	Depends on the process, and may depend on more arguments.
	"""
	pass

