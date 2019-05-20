import Dyadic_measures as dm
import Create_Graphs as graphs
from operator import mul
import time
import numpy as np
import itertools as itr
from numba import jit


#### TODO:
'''
Should Keep track of all of history in a list, in a different class: monitor Processes
'''



###############################################################################
###############################################################################
###############################################################################
### Class definition of Local approximation###


class LocalApproximationSimulation:
	'''
	'''
	def __init__(self, kappa, initial_measure = None, initial_history = 1):
		self.initial_history = initial_history
		self.dimension =  kappa + 1 # dimension should capture the degree of the graph + 1; 

		if initial_measure is not None:
			self.initial_measure = initial_measure
		else:
			self.initial_measure = dm.get_random_dyadic_measure(initial_history,self.dimension)


		self.current_measure = self.initial_measure
		self.previous_measure = None
		self.side_length = 2**self.initial_history# side_length of the measure
		self.history_length = self.initial_history
	

	def simulate_until_convergence(self, threshold = 1e-8):
		'''
		Simulate the transition operator until convergence
		'''
		consec_diff = []
		while True:
			diff = self.apply_nonlinear_markov_operator()
			consec_diff.append(diff)

			if consec_diff[-1] < threshold:
				break

		return (consec_diff, self.current_measure)



	def simulate_local_approximation(self, steps):
		'''
		Step forward with the transition operator
		'''
		#keep track of consecutive l2 differences

		consec_diff = []

		for i in range(steps):
			diff = self.apply_nonlinear_markov_operator(measure_diff = True)
			consec_diff.append(diff)

		return consec_diff

		


	# want to parallelize this + JIT.
	def apply_nonlinear_markov_operator(self, measure_diff):
		'''
		Should be the transition operator (string together the functions below)
		defines a function to update the current measure of the graph

		Returns the measure diff if paramtert is passed in , else None
		'''
		new_side_length = 2*self.side_length
		new_measure = np.zeros([new_side_length]*self.dimension)


		## Compute conditional expectations here
		cond_exp = self.compute_conditional_expectation()

		# update new measure by integrating the kernel over old measure
		# can probably parallellize this to make it faster

		for state in itr.product(range(self.side_length),repeat = self.dimension):
			print(state)
			#get the dictionary of where to map to with probabilities
			probability_kernel = self.get_probability_kernel(state, cond_exp)
			prob = self.current_measure[state]

			# update the new measure 
			new_measure = new_measure + probability_kernel*prob

		# measure the difference between the old and new measures
		diff = None
		if measure_diff:
			diff = np.sqrt( ((new_measure - dm.embed(self.current_measure,self.history_length+1))**2).sum())

		# update instance vars
		self.previous_measure = self.current_measure
		self.current_measure = new_measure
		self.side_length = new_side_length
		self.history_length += 1

		return diff


	def get_probability_kernel(self, state, cond_exp):
		'''
		Returns probability distribution of how to evolve at a given state.
		Returns P^mu(x, cdot) - 
		Params
		------
		state: coordinates of a vector with d dimensions

		------
		Output: returns the probability distribution in dict form; keys are the new states and 
		'''
		
		# get the product of bernoullis
		bernoulli_ps = np.array([self.get_bernoulli_functional(node,state,cond_exp) for node in range(self.dimension)])
		bernoulli_ps = np.array([1-bernoulli_ps,bernoulli_ps])

		new_side_length = 2*self.side_length
		probability_dict = np.zeros([new_side_length]*self.dimension)

		for transition in itr.product(range(2),repeat = self.dimension):

			probs = bernoulli_ps[transition,range(self.dimension)]

			# in the coordinates of the new measure, which has double the side length
			new_state = np.array(state) + self.side_length*np.array(transition) 

			probability_dict[tuple(new_state)] = np.prod(probs)


		return probability_dict


	def get_bernoulli_functional(self, node, state):
		'''
		Gets the probability of transitioning to state 1 for a given node.Should be model specific
		'''
		raise NotImplementedError

	def compute_conditional_expectation(self, node, state):
		'''
		Should be model specific
		'''
		raise NotImplementedError

###############################################################################
###############################################################################
###############################################################################
### The K Approximation is a subclass of the Local Approximation Simulation ###

class KApproximation(LocalApproximationSimulation):
	'''
	Class extends the Local Apporximation Simulation. The only further extension are the functions
	Simulate_k_approximation and Simulate_k_approximation_until_convergence
	'''
	def __init__(self,kappa,k = None, initial_measure = None, initial_history = 1):
		super(KApproximation, self).__init__(kappa, initial_measure, initial_history)
		self.history_length = initial_history
		if k is None:
			self.k = initial_history
		else:
			self.k = k


	def simulate_k_approximation(self, steps):
		'''
		Applies the smoothening operator to the simulate_local_approximation function
		'''
		consec_diff = []

		for i in range(steps):
			diff = self.apply_nonlinear_markov_operator(measure_diff = True)
			

			if self.history_length > self.k:
				smoothed_measure = dm.smoothen(self.current_measure,self.history_length - 1)

				consec_diff.append(np.sqrt(((smoothed_measure - self.previous_measure)**2).sum()))
				self.current_measure = smoothed_measure
				self.history_length -= 1
				self.side_length = 2**self.history_length
			else:
				consec_diff.append(diff)

		return consec_diff


	def simulate_k_approximation_until_convergence(self, threshold = 1e-8):
		'''
		Applies the smoothening operator to the simulate_local_approximation function
		'''
		consec_diff = []

		while True:
			self.apply_nonlinear_markov_operator(measure_diff = False)
			smoothed_measure = dm.smoothen(self.current_measure,self.history_length - 1)

			consec_diff.append(np.sqrt(((smoothed_measure - self.previous_measure)**2).sum()))
			self.current_measure = smoothed_measure
			self.history_length -= 1
			self.side_length = 2**self.history_length

			if consec_diff[-1] < threshold:
				break

		return (consec_diff,self.current_measure)

		
		

###############################################################################
###############################################################################
###############################################################################
### Miiscellaneous Helper Functions ###

def get_fcd(x,side_length):
	if isinstance(x,np.ndarray):
		return (2*x/side_length).astype(int)
	else:
		return int(2*x/side_length)



