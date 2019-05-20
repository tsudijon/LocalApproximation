################################################
################################################
################################################
### Functionals to analyze ###
# Functionals looking at marginals, vs functionals looking over a time step.

# Probabilities of a node being 1,or -1 at stationarity

# Occupation measure: fraction of time a node is at state 1 out of T timesteps.

#### Don't need to do the below for the Mean Field Simulation

# 2 nodes neighboring each other - get the joint distribution - at a large fixed time

# 2 nodes neighboring each other - look at joint distribution/ occupation measure.

################################################
import JittedIsingLocalApprox as Local
import Dyadic_measures as measures
import numpy as np
import functools
import Misc


class MonitorLocalApproximation():
	def __init__(self, initial_measure, p, beta):
		self.measure_history = [initial_measure]
		self.current_measure = initial_measure
		self.consec_diff = []
		self.p = p
		self.beta = beta


	def run(self,steps):
		for s in range(steps):
			(diff,measure) = Local.simulate_local_approximation(1,self.current_measure, self.p, self.beta)
			self.consec_diff.append(diff)
			self.measure_history.append(measure)
			self.current_measure = measure

	def get_node_marginal(self,nodes, T):
		"""
		gets the marginal of a node
		from the most current measure, up to time T
		returns a measure of their histories up to time T

		Params:
		---------------
		node: list of int
			(integers from 0 to d)
		T: int
			a time up until which to get the marginal
		"""
		current_measure = self.measure_history[T]
		num_nodes = len(current_measure.shape)

		assert max(nodes) <= num_nodes, "nodes should be less than d+1"
		assert len(nodes) == len(set(nodes)), "nodes should be unique"
		
		# sum over the other axes
		other_axes = tuple(set(range(num_nodes)) - set(nodes))
		joint_marginal = np.sum(current_measure,axis = other_axes)

		return joint_marginal

	def get_fixed_time_measure(self,nodes,T):
		"""
		Get the distribution of a given node at a fixed time T.

		Params:
		-----------------
		nodes: list of int
			integers between 0,...,d
		T: int
			time at which to get the distribution
		"""
		joint_marginal = self.get_node_marginal(nodes,T)

		return measures.smoothen(joint_marginal,1)

	def get_occupation_measure(self,nodes,T):
		"""
		Get the occupation measure of given nodes up until time T. - the amount of time
		the process spends in some state.

		In the case of the K-approximation, can only do this for a given length
		Returns a dictionary; keys are states and values are average times spent in that state

		Currently only works for processes with two values in the state space.

		Params:
		------------------
		nodes: list of int
		T: int
		"""
		occupation_dict = dict()
		joint_marginals = []

		# average over histories:
		for i in range(T):
			marginal_at_t = measures.smoothen(self.get_node_marginal(nodes,i),1)
			marginal_at_t = {key:value for key,value in np.ndenumerate(marginal_at_t)}
			joint_marginals.append(marginal_at_t)

		occupation_dict = functools.reduce(Misc.sum_dicts, joint_marginals)
		occupation_dict = {key:(value/T) for key,value in occupation_dict.items()}

		return occupation_dict

	def collect_statistics(self,stationary_T, small_T):
		"""
		Collects the following information from the process:
		Fixed T joint, neighborhing nodes. Occupation measure up to time T
		of neighboring nodes and a fixed node.

		Params:
		----------------
		stationary_T: int
			Large time at which to collect joint distribution of neighboring nodes
			along with occupation measure; mimicks stationarity
		small_T: int
			Time at which to collect joint distribution and occupation measure of
			single nodes and neighboring nodes
		"""
		pass


class MonitorKApproximation(MonitorLocalApproximation):
	def __init__(self,initial_measure,k):
		super(MonitorKApproximation,self).__init__(initial_measure)
		self.k = k

	def run(self,steps):
		for s in range(steps):
			(diff,measure) = Local.simulate_k_approximation(1,self.current_measure, self.k, self.p, self.beta)
			self.consec_diff.append(diff)
			self.measure_history.append(measure)
			self.current_measure = measure
















