import os 
import sys
sys.path.append("../SimulateProcesses/")
sys.path.append("../SimulateProcesses/Contact")
import Create_Graphs as graphs 
import Dyadic_measures as measures
import JittedContactProcess as Contact 
import ContactMonitor as monitor
import Misc

# import standard libraries
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

import ast
import importlib
import argparse
import networkx as nx


def run_k_approximation(depth,kappa, T,p ,q, k, bias, load_data = False):
	"""
	Runs the k approximation and computes the occupation measure and fixed measure
	of a node and of the joint between zero and a neighbor, say 1.


	Params
	-----------------------------
	depth: int
		depth of the tree
	kappa: int
		degree of the tree
	T: int
		length of returned sequence of measures: runs up until time T-1
	p: float in [0,1]
		Infection Rate
	q: float in [0,1]
		Recovery Rate
	k: int
		history length for the k approximation
	bias: float in [-1,1]
		affects the initial probability.
	"""

	## Create graphs and processes
	f = "ContactProcessResults/kapprox/kapprox_kappa{}_p{}_q{}_T{}_k{}_bias{}.npy".format(kappa,p,q,T,k,bias)

	# create a product measure given by bias
	biased_p = (1+bias)/2
	initial = np.zeros(tuple([2]*(kappa+1)))
	for idx in itr.product([0,1],repeat = (kappa+1)):
		initial[idx] = ((biased_p)**(sum(idx)))*((1-biased_p)**(kappa + 1 - sum(idx)))

	initial = measures.dyadic_measure(initial,k = 1, dimension = kappa + 1)

	process = monitor.MonitorKApproximation(initial,k,p,q)


	# load / create data
	if load_data == False:
		process.run(T-1)
		np.save(f,process.measure_history)
	else:
		measure_history = np.load(f)
		process.measure_history = measure_history


	### Extract statistics
	marginal_of_root = process.get_fixed_time_measure([0],T-1)
	marginal_of_nbr = process.get_fixed_time_measure([1],T-1)
	marginal_of_root_and_nbr = process.get_fixed_time_measure([0,1],T-1)

	# convert marginal of root and nbr into a dict
	marginal_of_root_and_nbr = {k:v for k,v in np.ndenumerate(marginal_of_root_and_nbr)}

	occ_measure_of_root = process.get_occupation_measure([0],T-1)
	occ_measure_of_nbr = process.get_occupation_measure([1],T-1)
	occ_measure_of_root_and_nbr = process.get_occupation_measure([0,1],T-1)

	root_history = [process.get_fixed_time_measure([0],i) for i in range(T)]
	joint_history = [process.get_fixed_time_measure([0,1],i) for i in range(T)]

	return (marginal_of_root, marginal_of_root_and_nbr, \
			occ_measure_of_root, occ_measure_of_root_and_nbr, root_history, joint_history)


def run_local_approximation(depth,kappa, T,p ,q, bias, load_data = False):
	"""
	Runs the local approximation and computes the occupation measure and fixed measure
	of a node and of the joint between zero and a neighbor, say 1. Time should be small.


	Params
	-----------------------------
	depth: int
		depth of the tree
	kappa: int
		degree of the tree
	T: int
		length of returned sequence of measures: runs up until time T-1
	p: float in [0,1]
		Infection Rate
	q: float in [0,1]
		Recovery Rate
	bias: float in [-1,1]
		initial probability  = (1 + bias) / 2
	load_data: Boolean
		if True, loads stored data
	"""

	f = "ContactProcessResults/localapprox/localapprox_kappa{}_p{}_q{}_T{}_bias{}.npy".format(kappa,p,q,T,bias)

	# create a product measure given by bias
	biased_p = (1+bias)/2
	initial = np.zeros(tuple([2]*(kappa+1)))
	for idx in itr.product([0,1],repeat = (kappa+1)):
		initial[idx] = ((biased_p)**(sum(idx)))*((1-biased_p)**(kappa + 1 - sum(idx)))

	initial = measures.dyadic_measure(initial,k = 1, dimension = kappa + 1)

	process = monitor.MonitorLocalApproximation(initial,p,q)

	# load / create data
	if load_data == False:
		process.run(T-1)
		np.save(f,process.measure_history)

	else:
		measure_history = np.load(f)
		process.measure_history = measure_history


	### extract statistics
	marginal_of_root = process.get_fixed_time_measure([0],T-1)
	marginal_of_nbr = process.get_fixed_time_measure([1],T-1)
	marginal_of_root_and_nbr = process.get_fixed_time_measure([0,1],T-1)

	# convert marginal of root and nbr into a dict
	marginal_of_root_and_nbr = {k:v for k,v in np.ndenumerate(marginal_of_root_and_nbr)}

	occ_measure_of_root = process.get_occupation_measure([0],T-1)
	occ_measure_of_nbr = process.get_occupation_measure([1],T-1)
	occ_measure_of_root_and_nbr = process.get_occupation_measure([0,1],T-1)

	root_history = [process.get_fixed_time_measure([0],i) for i in range(T)]
	joint_history = [process.get_fixed_time_measure([0,1],i) for i in range(T)]

	return (marginal_of_root, marginal_of_root_and_nbr, \
			occ_measure_of_root, occ_measure_of_root_and_nbr, root_history, joint_history)
	

def run_dynamics(depth, kappa, T, p, q, bias, bd_condition, load_data = False, use_bc = False):
	"""
	Gets the occupation measure up to time T of node 0 and the joint of node 0 and a neighboring node
	Also returns the distribution at a fixed time for a single node. This can be used for large T.
	Calculates these quantities via monte carlo simulation

	We automatically set the boundary to all +1, although the functionality accomodates for other boundary conditions

	We remove errorbars.

	Params
	-----------------------------
	depth: int
		depth of the tree
	kappa: int
		degree of the tree
	T: int
		Time to run until
	p: float in [0,1]
		Infection Rate
	q: positive float
		Recovery Rate
	bias: float in [-1,1]
		affects the starting probability via p_0 = (1 + bias) / 2
	bd_condition: list of +1,0
		boundary condition that is fixed
	load_data: boolean
		loads data if true
	use_bc: boolean
		if true, fixes boundary condition at supplied values throughout simulation
	"""

	## Create graphs and processes

	f = "ContactProcessResults/dynamics/dynamics_kappa{}_depth{}_p{}_q{}_T{}_bias{}".format(kappa,depth,p,q,T,bias)
	jointf = "ContactProcessResults/dynamics/joint_dynamics_kappa{}_depth{}_p{}_q{}_T{}_bias{}".format(kappa,depth,p,q,T,bias)
	
	N = 10000
		
	graph = graphs.create_regular_tree(kappa,depth)
	#graph = nx.complete_graph(depth)

	# get probabilities by averaging over histories
	joint_probs = Contact.simulate_Contact_Process_dynamics(graph,T,p,q,bd_condition,bias, N, use_bc)

	probs = np.zeros((2,T))
	probs[0,:] = joint_probs[[0,1],:].sum(axis = 0)
	probs[1,:] = joint_probs[[2,3],:].sum(axis = 0)


	## Return marginals of the root and neighbor 
	marginal_of_root = probs[0,T-1]
	marginal_of_root_and_nbr = joint_probs[:,T-1]

	marginal_of_root_and_nbr_dict = {}
	marginal_of_root_and_nbr_dict[(0,0)] = joint_probs[0,T-1]
	marginal_of_root_and_nbr_dict[(0,1)] = joint_probs[1,T-1]
	marginal_of_root_and_nbr_dict[(1,0)] = joint_probs[2,T-1]
	marginal_of_root_and_nbr_dict[(1,1)] = joint_probs[3,T-1]


	## For the occupation measure
	#for the marginal, can average over the probs.
	occ_measure_of_root = np.mean(probs[0,:])

	joint_occ = {}
	joint_occ[(0,0)] = np.mean(joint_probs[0,:])
	joint_occ[(0,1)] = np.mean(joint_probs[1,:])
	joint_occ[(1,0)] = np.mean(joint_probs[2,:])
	joint_occ[(1,1)] = np.mean(joint_probs[3,:])


	joint_probs_dicts = [0]*T
	for i in range(T):
		d = {}
		d[(0,0)] = joint_probs[0,i]
		d[(0,1)] = joint_probs[1,i]
		d[(1,0)] = joint_probs[2,i]
		d[(1,1)] = joint_probs[3,i]
		joint_probs_dicts[i] = str(d)
	joint_probs_dicts = np.array(joint_probs_dicts)

	return (marginal_of_root, marginal_of_root_and_nbr_dict, \
			 occ_measure_of_root,
			 joint_occ,
			  probs[0,:],
			   joint_probs_dicts)

def run_mean_field_simulations(kappa,p_0,T,p,q, load_data = False):
	"""
	Gets the occupation measure for the mean field approximation up to time T of node 0 and the joint of node 0 and a neighboring node
	Also returns the distribution at a fixed time for a single node. This can be used for large T.
	Calculates these by recursion.

	Params
	-----------------------------
	kappa: int
		degree of the tree
	p_0: float in [0,1]
		initial probability of node being 1
	T: int
		Time to run until
	p: float in [0,1]
		Infection Rate
	q: positive float
		Recovery Rate
	load_data: boolean
		if data is stored, use that instead
	"""

	f = "ContactProcessResults/meanfield/meanfield_kappa{}_p0{}_p{}_q{}_T{}.npy".format(kappa,p_0,p,q,T)
	if load_data == False:
		probs = Contact.simulate_mean_field_dynamics(kappa,p_0,T,p,q)
		np.save(f,probs)
	else:
		probs = np.load(f)
	print(probs)

	# just return the product measure for each
	def get_joint(prob):
		joint = {}
		joint[(1,1)] = np.power(prob,2)
		joint[(1,0)] = prob*(1-prob)
		joint[(0,1)] = (1-prob)*prob
		joint[(0,0)] = np.power((1-prob),2)
		return joint 

	## get joint measure for node and its neighbor
	joint_probs = [get_joint(prob) for prob in probs]

	## get the occupation measure (expected amount of time at 1)
	occ_measure = sum(probs)/T

	## get the joint occupation measure
	joint_occ_measure = {k:(v/T) for k,v in functools.reduce(Misc.sum_dicts,joint_probs).items()}

	return (probs[T-1], joint_probs[T-1], occ_measure, joint_occ_measure, probs, joint_probs)
	
#########################################################################################################
#########################################################################################################
#########################################################################################################
### Plotting Code ###

def plot_vs_kappa(df, kappa_params, depth_params, k_params, title, filename, scale_y = False, show_plot = False, yaxis = 'Probability'):
	"""
	Plots dataframe of probabilities against increasing values of kappa.

	Params
	-------
	df: dataframe
		df generated in simulations, records probabilities as a function of kappa and depth
	kappa_params: list
		list of kappa params for the df
	depth_params: list
		list of depth params chosen for datafrane
	k_params: list
		list of k values for the k approximation
	title: string
		title for plot
	filename: string
		where to save the figure to
	scale_y: Boolean
		scales the y axis to 0,1 if true
	show_plot: Boolean
		shows the plot if true
	yaxis: string
		title for yaxis
	"""
	plt.figure(figsize = (5,5))
	ax = plt.gca()

	idx = pd.IndexSlice

	mf_vals = df.loc[idx[kappa_params,depth_params[0]],'Mean Field']
	ax.plot(kappa_params, mf_vals, marker = 'o', label = "Mean Field")

	for d in depth_params:
		dynamics_vals = df.loc[idx[kappa_params,d],'Dynamics']
		ax.plot(kappa_params, dynamics_vals, marker = 'o', label = "Dynamics depth {}".format(d))

	for k in k_params:
		kapprox_vals = df.loc[idx[kappa_params,depth_params[0]],'KApprox_{}'.format(k)]
		ax.plot(kappa_params, kapprox_vals, marker = 'o', label = "{}-Approximation".format(k))


	plt.xlabel('Kappa')
	plt.ylabel(yaxis)
	plt.xticks(kappa_params)
	if scale_y == True:
		plt.yticks(np.arange(0,1.1,0.1))
		plt.ylim(0,1)
	plt.legend()
	plt.title(title)
	plt.savefig(filename)
	if show_plot:
		plt.show()

def plot_joint_vs_kappa(df, kappa_params, depth_params, k_params, title, state, filename, scale_y = False, show_plot = False, yaxis = 'Probability'):
	"""
	Plots dataframe of probabilities against increasing values of kappa.

	Params
	-------
	df: dataframe
		df generated in simulations, records probabilities as a function of kappa and depth
	kappa_params: list
		list of kappa params for the df
	depth_params: list
		list of depth params chosen for datafrane
	k_params: list
		list of k values for the k approximation
	title: string
		title for plot
	filename: string
		where to save the figure to
	scale_y: Boolean
		scales the y axis to 0,1 if true
	show_plot: Boolean
		shows the plot if true
	yaxis: string
		title for yaxis
	"""
	plt.figure(figsize = (5,5))
	ax = plt.gca()

	def str_to_dict(str):
		d = ast.literal_eval(str)
		return d[state]

	idx = pd.IndexSlice


	mf_vals = df.loc[idx[kappa_params,depth_params[0]],'Mean Field'].values
	mf_vals = list(map(str_to_dict,mf_vals))
	ax.plot(kappa_params, mf_vals, marker = 'o', label = "Mean Field")

	for d in depth_params:
		dynamics_vals = df.loc[idx[kappa_params,d],'Dynamics'].values
		dynamics_vals = list(map(str_to_dict,dynamics_vals))
		ax.plot(kappa_params, dynamics_vals, marker = 'o', label = "Dynamics depth {}".format(d))

	for k in k_params:
		kapprox_vals = df.loc[idx[kappa_params,depth_params[0]],'KApprox_{}'.format(k)].values
		kapprox_vals = list(map(str_to_dict,kapprox_vals))
		ax.plot(kappa_params, kapprox_vals, marker = 'o', label = "{}-Approximation".format(k))


	plt.xlabel('Kappa')
	plt.ylabel(yaxis)
	plt.xticks(kappa_params)
	if scale_y == True:
		plt.yticks(np.arange(0,1.1,0.1))
		plt.ylim(0,1)
	plt.legend()
	plt.title(title)
	plt.savefig(filename)
	if show_plot:
		plt.show()

def plot_prob_vs_time(df, kappa, depth, k_params, title, filename, scale_y = False, show_plot = False):
	"""
	Plots dataframe of single node probabilities against time for fixed values of kappa and depth

	Params
	-------
	df: dataframe
		df generated in simulations, records probabilities as a function of kappa and depth
	kappa: int
		kappa param to plot
	depth: int
		depth param to plot 
	k_params: list
		list of k values for the k approximation
	title: string
		title for plot
	filename: string
		where to save the figure to
	scale_y: Boolean
		scales the y axis to 0,1 if true
	show_plot: Boolean
		shows the plot if true
	"""
	idx = pd.IndexSlice
	plt.figure(figsize = (5,5))
	ax = plt.gca()

	T = len(df)

	mf_vals = df.loc[:,idx['Mean Field',kappa,depth]]
	ax.plot(range(T), mf_vals, label = "Mean Field")

	dynamics_vals = df.loc[:,idx['Dynamics',kappa, depth]]
	ax.plot(range(T), dynamics_vals, label = "Dynamics")

	for k in k_params:
		kapprox_vals = df.loc[:,idx['KApprox_{}'.format(k),kappa, depth]] 
		ax.plot(range(T), kapprox_vals, label = "{}-Approximation".format(k))


	plt.xlabel('Time')
	plt.ylabel('Probability')
	if scale_y == True:
		plt.yticks(np.arange(0,1.1,0.1))
		plt.ylim(0,1)
	plt.legend()
	plt.title('{}, Kappa {}, Depth {}'.format(title,kappa,depth))
	plt.savefig(filename)
	if show_plot:
		plt.show()

def plot_joint_state_prob_vs_time(df, kappa, depth, k_params, state, title, filename, scale_y = False, show_plot = False):
	"""
	Plots dataframe of joint probabilities against time for fixed values of kappa and depth

	Params
	-------
	df: dataframe
		df generated in simulations, records probabilities as a function of kappa and depth
	kappa: int
		kappa param to plot
	depth: int
		depth param to plot 
	k_params: list
		list of k values for the k approximation
	state: str
		state of root and neighbor of joint probabilities
	title: string
		title for plot
	filename: string
		where to save the figure to
	scale_y: Boolean
		scales the y axis to 0,1 if true
	show_plot: Boolean
		shows the plot if true
	"""

	plt.figure(figsize = (5,5))
	ax = plt.gca()

	def str_to_dict(str):
		d = ast.literal_eval(str)
		return d[state]

	T = len(df)
	idx = pd.IndexSlice
	mf_vals = df.loc[:,idx['Mean Field',kappa,depth]].values
	mf_vals = list(map(str_to_dict,mf_vals))
	ax.plot(range(T), mf_vals, label = "Mean Field")

	dynamics_vals = df.loc[:,idx['Dynamics',kappa, depth]].values
	dynamics_vals = list(map(str_to_dict,dynamics_vals))
	ax.plot(range(T), dynamics_vals, label = "Dynamics")

	for k in k_params:
		kapprox_vals = df.loc[:,idx['KApprox_{}'.format(k),kappa, depth]].values
		kapprox_vals = list(map(str_to_dict,kapprox_vals))
		ax.plot(range(T), kapprox_vals, label = "{}-Approximation".format(k))


	plt.xlabel('Time')
	plt.ylabel('Probability')
	if scale_y == True:
		plt.yticks(np.arange(0,1.1,0.1))
		plt.ylim(0,1)
	plt.legend()
	plt.title('{}, Kappa {}, Depth {}, State {}'.format(title,kappa,depth,state))
	plt.savefig(filename)
	if show_plot:
		plt.show()