import os 
import sys


# import standard libraries
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

import ast
import importlib

from numba import jit, njit

###########################################
###########################################
###########################################
### dyadic measures code ###

def smoothen(measure,k):
	'''
	Input: a dyadic measure. 
	Output: a dyadic measure of side length k.
	'''
	side_len = measure.shape[0]
	dim = len(measure.shape)

	assert side_len >= 2**k

	if side_len == 2**k:
		return measure

	smoothed_measure = np.zeros(shape = tuple([2**k]*dim))
	smoothed_idxs = np.arange(0,2**k)
	smoothed_len = int(side_len/(2**k))

	for cds in itr.product(smoothed_idxs,repeat = dim):

		smaller_cube = [range(smoothed_len*cd, smoothed_len*(cd+1)) for cd in cds]
		# * is the splat operator: turns a list of arguments into inputs for an arg
		smaller_cube_cds = [list(tp) for tp in itr.product(*smaller_cube)] # n d-tuples
		coordinate_lists = list(np.array(smaller_cube_cds).T) # d lists of n coordinates
		smoothed_measure[cds] = measure[coordinate_lists].sum() #
        
	return smoothed_measure

def embed(measure,k):
	'''
	View a measure as one with greater side length
	'''
	side_len = measure.shape[0]
	dim = len(measure.shape)

	assert side_len <= 2**k

	if side_len == 2**k:
		return measure

	refined_measure = np.zeros(shape = tuple([2**k]*dim))
	refined_idxs = np.arange(0,2**k)
	refined_len = int((2**k)/side_len)

	for cds in itr.product(refined_idxs,repeat = dim):
		cds_in_old_measure = (np.array(cds)/refined_len).astype(int)
		refined_measure[cds] = measure[tuple(cds_in_old_measure)]/(refined_len**dim)

	return refined_measure


###########################################
###########################################
###########################################

###########################################
###########################################
###########################################
### Code for Local Approximation ###


def simulate_local_approximation(steps, measure, p, beta):
	'''
	Step forward with the transition operator
	'''
	#keep track of consecutive l2 differences

	consec_diff = []

	for i in range(steps):
		(diff,new_measure) = apply_nonlinear_markov_operator(True, measure, p, beta)
		consec_diff.append(diff)

	return (consec_diff,new_measure)

def apply_nonlinear_markov_operator(measure_diff, current_measure, p, beta):
	'''
	Should be the transition operator (string together the functions below)
	defines a function to update the current measure of the graph

	Returns the measure diff if paramtert is passed in , else None
	'''
	side_length = current_measure.shape[0]
	dimension = len(current_measure.shape)
	history_length = int(np.log2(side_length))

	new_side_length = 2*side_length
	new_measure = np.zeros([new_side_length]*dimension)


	## Compute conditional expectations here
	cond_exp = compute_conditional_expectation(side_length, current_measure, p, beta)

	# update new measure by integrating the kernel over old measure
	# can probably parallellize this to make it faster

	args = tuple([list(range(side_length))]*dimension)
	for state in np.array(np.meshgrid(*args)).T.reshape(-1,dimension):
		print(tuple(state))
		#get the dictionary of where to map to with probabilities
		probability_kernel = get_probability_kernel(tuple(state), cond_exp, dimension, side_length, p, beta)
		prob = current_measure[tuple(state)]

		# update the new measure 
		new_measure = new_measure + probability_kernel*prob

	# measure the difference between the old and new measures
	diff = None
	if measure_diff:
		diff = np.sqrt( ((new_measure - embed(current_measure, history_length +1))**2).sum())

	return (diff,new_measure)


#@njit(cache = True)
def get_probability_kernel(state, cond_exp, dimension, side_length, p, beta):
	'''
	Returns probability distribution of how to evolve at a given state.
	Returns P^mu(x, cdot) - 
	Params
	------
	state: coordinates of a vector with d dimensions

	------
	Output: returns the probability distribution in dict form; keys are the new states and 
	'''
	get_bernoulli_functional(0, state, cond_exp, side_length, p, beta)
	
	# get the product of bernoullis
	bernoulli_ps = np.array([get_bernoulli_functional(node, state, cond_exp, side_length, p, beta) for node in range(dimension)])
	bernoulli_ps = np.array([1-bernoulli_ps,bernoulli_ps])

	new_side_length = 2*side_length
	probability_dict = np.zeros([new_side_length]*dimension)

	args = tuple([[0,1]]*dimension)
	for transition in np.array(np.meshgrid(*args)).T.reshape(-1,dimension):

		probs = bernoulli_ps[tuple(transition),range(dimension)]

		# in the coordinates of the new measure, which has double the side length
		new_state = np.array(state) + side_length*np.array(transition) 

		probability_dict[tuple(new_state)] = np.prod(probs)

	return probability_dict

@njit(cache=True)
def get_bernoulli_functional(node, state, cond_exp, side_length, p, beta):
	'''
	get the probability of being 1 at the next time step for each node, given the state of the process.

	'''

	# return the ordinary probability
	if node == 0:

		current_sign = int(2*((state[node]))/side_length)

		functional = probability_of_one(current_sign, p, beta,
						np.sum( 2 * ((2*np.array(state[1:])/side_length).astype(np.int32) )  -1) )

	else:
		functional = cond_exp[state[node],state[0]]


	return functional

@jit(cache=True)
def get_fcd(x,side_length):
	if isinstance(x,np.ndarray):
		return (2*x/side_length).astype(int)
	else:
		return int(2*x/side_length)

@njit(cache=True)
def compute_conditional_expectation(side_length, current_measure, p, beta):
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

			# compute the expectation by looping through indices of the conditional measure
			weighted_values = [prob*probability_of_one(current_sign, p, beta, 
							np.sum( 2*( (2*np.array(idx[1:])/side_length).astype(np.int32) ) - 1 )) for idx, prob in \
								np.ndenumerate(current_measure) if in_conditional_measure(idx)]	

			conditional_expectations[i,j] = np.sum(np.array(weighted_values,dtype = np.float64))

	return conditional_expectations

@njit(cache=True)
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


