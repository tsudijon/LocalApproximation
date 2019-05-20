import os 
import sys
sys.path.append("../SimulateProcesses/")
sys.path.append("../SimulateProcesses/Oscillator")
import Create_Graphs as graphs 
import Dyadic_measures as measures
import Automata as automata 
import AutomataMonitor as monitor
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


def run_k_approximation(depth,kappa, T, p, k, bias, load_data = False):
	"""
	Runs the k approximation and computes the occupation measure and fixed measure
	of a node and of the joint between zero and a neighbor, say 1.

	TODO: make this take in a measure


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
	q: float in [0,1]
		Recovery Rate
	k: int
		history length for the k approximation
	"""

	## Create graphs and processes
	f = "AutomataResults/kapprox/kapprox_kappa{}_p{}_T{}_k{}_bias{}.npy".format(kappa,p,T,k,bias)

	# create a product measure given by bias
	biased_p = (1+bias)/2
	initial = np.zeros(tuple([2]*(kappa+1)))
	for idx in itr.product([0,1],repeat = (kappa+1)):
		initial[idx] = ((biased_p)**(sum(idx)))*((1-biased_p)**(kappa + 1 - sum(idx)))

	initial = measures.dyadic_measure(initial,k = 1, dimension = kappa + 1)

	process = monitor.MonitorKApproximation(initial,k,p)


	if load_data == False:
		## Analyze Process 
		process.run(T-1)

		np.save(f,process.measure_history)
	else:
		measure_history = np.load(f)
		process.measure_history = measure_history


	
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


def run_local_approximation(depth,kappa, T,p, bias, load_data = False):
	"""
	Ideally T here should be small.
	"""
	## Create graphs and processes

	## Analyze Process 

	f = "AutomataResults/localapprox/localapprox_kappa{}_p{}_T{}_bias{}.npy".format(kappa,p, T,bias)

	# create a product measure given by bias
	biased_p = (1+bias)/2
	initial = np.zeros(tuple([2]*(kappa+1)))
	for idx in itr.product([0,1],repeat = (kappa+1)):
		initial[idx] = ((biased_p)**(sum(idx)))*((1-biased_p)**(kappa + 1 - sum(idx)))

	initial = measures.dyadic_measure(initial,k = 1, dimension = kappa + 1)

	process = monitor.MonitorLocalApproximation(initial,p)


	if load_data == False:
		## Analyze Process 
		process.run(T-1)

		np.save(f,process.measure_history)
	else:
		measure_history = np.load(f)
		process.measure_history = measure_history


	
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
	

def run_dynamics(depth, kappa, T, p, bias, load_data = False):
	"""
	Gets the occupation measure up to time T of node 0 and the joint of node 0 and a neighboring node
	Also returns the distribution at a fixed time for a single node. This can be used for large T.
	Calculates these quantities via monte carlo simulation

	We automatically set the boundary to all +1, although the functionality accomodates for other boundary conditions

	TODO: Need to worry about correpsondence of initial measures, and take in different initial 
	measures

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
	fixed_time:
		time at which to measure stuff
	"""

	## Create graphs and processes

	f = "AutomataResults/dynamics/dynamics_kappa{}_depth{}_p{}_T{}_bias{}".format(kappa,depth,p,T,bias)

	jointf = "AutomataResults/dynamics/joint_dynamics_kappa{}_depth{}_p{}_T{}_bias{}".format(kappa,depth,p,T,bias)
	
	N = 5000
	if load_data == False:

		graph = graphs.create_regular_tree(depth = depth, kappa = kappa)
		# get probabilities by averaging over histories
		probs = automata.simulate_contrarian_voter_dynamics(graph, T, p, bias)

		# keep track of joint measure
		joint_dict = {k:v for k,v in zip(itr.product([0,1],[0,1]),[0]*4) }
		joint_measure = [joint_dict.copy()]*len(probs)

		# turn the sequence of histories into the right form
		for i in range(len(probs)):
			joint_state = tuple([probs[i][v] for v in [0,1]])
			updated_measure = joint_measure[i].copy()
			updated_measure[joint_state] += 1 

			# turn the values into lists
			for key in updated_measure.keys():
				updated_measure[key] = [updated_measure[key]]									
			joint_measure[i] = updated_measure

		# turn the vlues of probs into a list
		for i in range(len(probs)):
			probs[i] = {k:[v] for k,v in probs[i].items()}

		# average over different runs
		for i in range(N-1):
			print(i)
			dynamics = automata.simulate_contrarian_voter_dynamics(graph, T, p, bias)
			for j in range(len(probs)):
				probs[j] = Misc.append_dicts(probs[j],dynamics[j]) 							#append here

				# update joint measure as well
				joint_state = tuple([dynamics[j][v] for v in [0,1]])
				updated_measure = joint_measure[j].copy()

				for key in updated_measure.keys():
					updated_measure[key].append(0)
				updated_measure[joint_state][-1] = 1

				joint_measure[j] = updated_measure

		# get the sum of a list & the variance as well
		prob_means = [{k:(np.mean(v)) for k,v in d.items()} for d in probs] 	
		prob_std = 	[{k:(np.std(v)/np.sqrt(N)) for k,v in d.items()} for d in probs] 	

		joint_means = [{k:(np.mean(v)) for k,v in d.items()} for d in joint_measure]
		joint_std = [{k:(np.std(v)/np.sqrt(N)) for k,v in d.items()} for d in joint_measure]

		## save this data? #save the probs list
		#np.save(f + "raw_prob.npy",probs)
		#np.save(jointf + "raw_prob.npy",joint_measure)


		np.save(f + ".npy",prob_means)
		np.save(jointf + ".npy",joint_means)

		np.save(f + "_std.npy",prob_std)
		np.save(jointf + "_std.npy", joint_std)

	else:
		probs = np.load(f + "raw_prob.npy")
		joint_measure = np.load(jointf + "raw_prob.npy")

		prob_means = np.load(f + ".npy")
		joint_means = np.load(jointf + ".npy")

		prob_std = np.load(f + "_std.npy")
		joint_std = np.load(jointf + "_std.npy")


	## Return marginals of the root and neighbor 
	marginal_of_root = [d[0] for d in prob_means][T-1]
	root_std = [d[0] for d in prob_std][T-1]

	marginal_of_root_and_nbr = joint_means[T-1]
	root_and_nbr_std = joint_std[T-1]

	## For the occupation measure
	#for the marginal, can average over the probs.
	occ_measure_of_root = [np.mean([d[0][i] for d in probs]) for i in range(N)]

	joint_occ = { k:([np.mean([d[k][i] for d in joint_measure]) for i in range(N)]) for k in joint_measure[0].keys()}

	mean_occ_joint = {k:np.mean(joint_occ[k]) for k in joint_occ.keys()}

	std_occ_joint = {k:(np.std(joint_occ[k])/np.sqrt(N)) for k in joint_occ.keys()}

	return (marginal_of_root, root_std, marginal_of_root_and_nbr, root_and_nbr_std, \
			 np.mean(occ_measure_of_root),np.std(occ_measure_of_root)/np.sqrt(N),
			 mean_occ_joint, std_occ_joint ,
			  [d[0] for d in prob_means], [d[0] for d in prob_std],
			   joint_means, joint_std)

def run_mean_field_simulations(kappa,p_0,T,p, load_data = False):

	f = "AutomataResults/meanfield/meanfield_kappa{}_p0{}_p{}_T{}.npy".format(kappa,p_0,p,T)
	if load_data == False:
		probs = automata.simulate_mean_field_dynamics(T, kappa, p, p_0)
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

def plot_vs_kappa(df,error_df, kappa_params, depth_params, k_params, title, filename, scale_y = False, show_plot = False):
	plt.figure(figsize = (5,5))
	ax = plt.gca()

	idx = pd.IndexSlice

	mf_vals = df.loc[idx[kappa_params,depth_params[0]],'Mean Field']
	ax.plot(kappa_params, mf_vals, marker = 'o', label = "Mean Field")

	for d in depth_params:
		dynamics_vals = df.loc[idx[kappa_params,d],'Dynamics']
		errors = error_df.loc[idx[kappa_params,d],'Dynamics']
		ax.errorbar(kappa_params, dynamics_vals, yerr = errors, marker = 'o', label = "Dynamics depth {}".format(d))

	for k in k_params:
		kapprox_vals = df.loc[idx[kappa_params,depth_params[0]],'KApprox_{}'.format(k)]
		ax.plot(kappa_params, kapprox_vals, marker = 'o', label = "{}-Approximation".format(k))


	plt.xlabel('Kappa')
	plt.ylabel('Probability')
	plt.xticks(kappa_params)
	if scale_y == True:
		plt.yticks(np.arange(0,1.1,0.1))
		plt.ylim(0,1)
	plt.legend()
	plt.title(title)
	plt.savefig(filename)
	if show_plot:
		plt.show()

def plot_joint_vs_kappa(df,error_df, kappa_params, depth_params, k_params, title, state, filename, scale_y = False, show_plot = False):

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
		errors = error_df.loc[idx[kappa_params,d],'Dynamics'].values
		errors = list(map(str_to_dict,errors))
		ax.errorbar(kappa_params, dynamics_vals,yerr = errors, marker = 'o', label = "Dynamics depth {}".format(d))

	for k in k_params:
		kapprox_vals = df.loc[idx[kappa_params,depth_params[0]],'KApprox_{}'.format(k)].values
		kapprox_vals = list(map(str_to_dict,kapprox_vals))
		ax.plot(kappa_params, kapprox_vals, marker = 'o', label = "{}-Approximation".format(k))


	plt.xlabel('Kappa')
	plt.ylabel('Probability')
	plt.xticks(kappa_params)
	if scale_y == True:
		plt.yticks(np.arange(0,1.1,0.1))
		plt.ylim(0,1)
	plt.legend()
	plt.title(title)
	plt.savefig(filename)
	if show_plot:
		plt.show()

def plot_prob_vs_time(df,error_df, kappa, depth, k_params, title, filename, scale_y = False, show_plot = False):
	idx = pd.IndexSlice
	plt.figure(figsize = (5,5))
	ax = plt.gca()

	mf_vals = df.loc[:,idx['Mean Field',kappa,depth]]
	ax.plot(range(T), mf_vals, label = "Mean Field")

	dynamics_vals = df.loc[:,idx['Dynamics',kappa, depth]]
	errors = error_df.loc[:,idx[kappa, depth]]
	ax.errorbar(range(T), dynamics_vals,yerr = errors, label = "Dynamics")

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

def plot_joint_state_prob_vs_time(df, error_df, kappa, depth, k_params, state, title, filename, scale_y = False, show_plot = False):

	plt.figure(figsize = (5,5))
	ax = plt.gca()

	def str_to_dict(str):
		d = ast.literal_eval(str)
		return d[state]


	idx = pd.IndexSlice
	mf_vals = df.loc[:,idx['Mean Field',kappa,depth]].values
	mf_vals = list(map(str_to_dict,mf_vals))
	ax.plot(range(T), mf_vals, label = "Mean Field")

	dynamics_vals = df.loc[:,idx['Dynamics',kappa, depth]].values
	dynamics_vals = list(map(str_to_dict,dynamics_vals))
	errors = error_df.loc[:,idx[kappa, depth]].values
	errors = list(map(str_to_dict,errors))
	ax.errorbar(range(T), dynamics_vals,yerr = errors, label = "Dynamics")

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




if __name__== "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()

	filename = args.filename

	np.random.seed(17)
	## at stationarity, which method works better?

	# Contact Process, below critical, above critical, at critical
	p = 0.6 # param for the automata
	bias = -0.2
	p_0 = (1+bias)/2

	## intake what parameters to run/ analyze
	kappa_params = [2,3,4,5,6]
	depth_params = [1,2,3,4,5]
	k_params = [1,2]

	# ideally want this to be larger
	T = 50

	### Collect Data
	# use Pandas, want multidimensional index; 
	# columns should be dynamics; mean field; k-approximation,various vals of k.

	columns = ["Dynamics","Mean Field"] + ["KApprox_{}".format(k) for k in k_params]
	idx_iter = [kappa_params,depth_params]
	index = pd.MultiIndex.from_product(idx_iter, names = ['kappa','depth'])
	zero_df = np.zeros((len(kappa_params)*len(depth_params),len(columns)))

	# probability at node 0, time T 
	single_node_prob_df = pd.DataFrame(zero_df,columns = columns, index = index)

	# joint probabilities of (0,1), time T
	joint_prob_df = single_node_prob_df.copy()

	# expected amount of time spent at 1, node 0
	single_node_occ_df = single_node_prob_df.copy()

	# expected amount of time spent by nodes (0,1) 
	joint_occ_df = single_node_prob_df.copy()
	
	### Store full history data ###
	idx = range(T)
	cols = ['Mean Field'] + ["KApprox_{}".format(k) for k in k_params] + ['Dynamics']
	colnames = [cols, kappa_params, depth_params]
	colnames = pd.MultiIndex.from_product(colnames,names = ['Type','Kappa','Depth'])
	zero_df= np.zeros((T,len(colnames)))
	single_node_history_df = pd.DataFrame(zero_df, columns = colnames, index = idx)
	joint_history_df = pd.DataFrame(zero_df, columns = colnames, index = idx)

	### Store the error bars for the dynamics
	columns = ["Dynamics"]
	idx_iter = [kappa_params,depth_params]
	index = pd.MultiIndex.from_product(idx_iter, names = ['kappa','depth'])
	zero_df = np.zeros((len(kappa_params)*len(depth_params),len(columns)))

	single_node_prob_error_df = pd.DataFrame(zero_df,columns = columns, index = index)
	joint_prob_error_df = pd.DataFrame(zero_df,columns = columns, index = index)
	single_node_occ_error_df = single_node_prob_error_df.copy()
	joint_occ_error_df = pd.DataFrame(zero_df,columns = columns, index = index)

	### Store error histories
	idx = range(T)
	colnames = [kappa_params,depth_params]
	colnames = pd.MultiIndex.from_product(colnames,names = ['Kappa','Depth'])
	zero_df= np.zeros((T,len(colnames)))
	single_node_error_history_df = pd.DataFrame(zero_df, columns = colnames, index = idx)
	joint_error_history_df = pd.DataFrame(zero_df, columns = colnames, index = idx)


	### run ordinary dynamics

	idx = pd.IndexSlice
	print("Running dynamics")
	for kappa, depth in itr.product(kappa_params,depth_params):
		print(kappa,depth)


		(root, root_std, joint, joint_std, root_occ, root_occ_std, joint_occ,
		 joint_occ_std, root_history,root_std_history, joint_history, joint_std_history) = \
			run_dynamics(depth,kappa,T,p,bias,load_data = False)

		single_node_prob_df.loc[idx[kappa,depth],idx['Dynamics']] = root
		joint_prob_df.loc[idx[kappa,depth],idx['Dynamics']] = str(joint)
		single_node_occ_df.loc[idx[kappa,depth],idx['Dynamics']] = root_occ
		joint_occ_df.loc[idx[kappa,depth],idx['Dynamics']] = str(joint_occ)

		# store the histories
		single_node_history_df.loc[:,idx['Dynamics',kappa,depth]] = root_history[:T] # there might be off by one error, check this out
		joint_history_df.loc[:,idx['Dynamics',kappa,depth]] = [str(d) for d in joint_history][:T]

		# store the errors
		single_node_prob_error_df.loc[idx[kappa,depth],idx['Dynamics']] = root_std
		joint_prob_error_df.loc[idx[kappa,depth],idx['Dynamics']] = str(joint_std)
		single_node_occ_error_df.loc[idx[kappa,depth],idx['Dynamics']] = root_occ_std
		joint_occ_error_df.loc[idx[kappa,depth],idx['Dynamics']] = str(joint_occ_std)
		# store error histories
		single_node_error_history_df.loc[:,idx[kappa,depth]] = root_std_history[:T] 
		joint_error_history_df.loc[:,idx[kappa,depth]] = [str(d) for d in joint_std_history][:T]



	## run mean field and get desired data
	print("Running mean field")
	for kappa in kappa_params:
		print("{}".format(kappa))
		(root, joint, root_occ, joint_occ, root_history, joint_history) = \
			run_mean_field_simulations(kappa,p_0,T,p, load_data = False)

		single_node_prob_df.loc[idx[kappa,:],idx['Mean Field']] = root
		joint_prob_df.loc[idx[kappa,:],idx['Mean Field']] = str(joint)
		single_node_occ_df.loc[idx[kappa,:],idx['Mean Field']] = root_occ
		joint_occ_df.loc[idx[kappa,:],idx['Mean Field']] = str(joint_occ)

		# store the histories
		hist = np.array(root_history[:T]).reshape((T,1))
		single_node_history_df.loc[:,idx['Mean Field',kappa]] = np.broadcast_to(hist,(T,len(depth_params)))
		joint_hist = np.array([str(d) for d in joint_history][:T]).reshape((T,1))
		joint_history_df.loc[:,idx['Mean Field',kappa]] = np.broadcast_to(joint_hist,(T,len(depth_params)))


	## run k approximation
	for k,kappa in itr.product(k_params,kappa_params):
		print("Running {}-Approx".format(k))
		print(kappa)


		(root, joint, root_occ, joint_occ, root_history, joint_history) = \
			run_k_approximation(1, kappa, T, p, k, bias, load_data = False) 

		single_node_prob_df.loc[idx[kappa,:],idx['KApprox_{}'.format(k)]] = root[1]
		joint_prob_df.loc[idx[kappa,:],idx['KApprox_{}'.format(k)]] = str(joint)
		single_node_occ_df.loc[idx[kappa,:],idx['KApprox_{}'.format(k)]] = list(root_occ.values())[1] #check this?
		joint_occ_df.loc[idx[kappa,:],idx['KApprox_{}'.format(k)]] = str(joint_occ)

		# store the histories
		h = np.array([a[1] for a in root_history[:T] ]).reshape((T,1))
		single_node_history_df.loc[:,idx['KApprox_{}'.format(k),kappa]] =  np.broadcast_to(h,(T,len(depth_params)))

		def array_to_dict(array):
			keys = []
			vals = []
			for k,v in np.ndenumerate(array):
				keys.append(k)
				vals.append(v)
			return dict(zip(keys, vals))

		joint_h = np.array([str(array_to_dict(d)) for d in joint_history][:T]).reshape((T,1))
		joint_history_df.loc[:,idx['KApprox_{}'.format(k),kappa]] = np.broadcast_to(joint_h,(T,len(depth_params)))

	# save the dataframes
	def get_df_string(s):
		f = "AutomataResults/dataframes/{}_T{}_maxkappa{}_maxdepth{}_k{}".format(s,T,
				max(kappa_params), max(depth_params), max(k_params))
		return f
	
	single_node_prob_df.to_pickle(get_df_string("single_node_prob_df"))
	joint_prob_df.to_pickle(get_df_string("joint_prob_df"))
	single_node_occ_df.to_pickle(get_df_string("single_node_occ_df"))
	joint_occ_df.to_pickle(get_df_string("joint_occ_df"))

	single_node_prob_error_df.to_pickle(get_df_string("single_node_prob_error_df"))
	joint_prob_error_df.to_pickle(get_df_string("joint_prob_error_df"))
	single_node_occ_error_df.to_pickle(get_df_string("single_node_occ_error_df"))
	joint_occ_error_df.to_pickle(get_df_string("joint_occ_error_df"))

	######################################################
	### Plots ###
	## function of kappa
	
	# single node probabilities
	file = "{}_{}_p{}_bias{}.jpg".format(filename,"root_prob", p, bias)
	plot_vs_kappa(single_node_prob_df, single_node_prob_error_df, kappa_params, depth_params[1:],
		k_params, "Root Node Probability of 1", file, False, False)

	# occupation measure
	file = "{}_{}_p{}_bias{}.jpg".format(filename,"root_occ", p, bias)
	plot_vs_kappa(single_node_occ_df, single_node_occ_error_df, kappa_params, depth_params[1:],
		k_params, "Root Node, Expected Time at State 1", file, False, False)

	## for single node results, plot as a function of T
	for kappa in kappa_params:
		for depth in depth_params:
			file = "{}_{}_p{}_bias{}_kappa{}_depth{}.jpg".format(filename,"root_prob_vs_time", p, bias, kappa, depth)
			plot_prob_vs_time(single_node_history_df,single_node_error_history_df, kappa,depth,k_params, "Automata", file, False, False)

	# joint probability, for a given state
	for state in [(0,0),(0,1),(1,0),(1,1)]:
		file = "{}_{}_p{}_bias{}_state{}.jpg".format(filename,"joint_prob",
										 p, bias,state)

		plot_joint_vs_kappa(joint_prob_df, joint_prob_error_df, kappa_params, 
			depth_params, k_params,"Probability of State {}".format(state), state, file, False, False)

		file = "{}_{}_p{}_bias{}_state{}.jpg".format(filename,"joint_occ",
										 p, bias, state)

		# joint occupation measure, for a given state
		plot_joint_vs_kappa(joint_occ_df, joint_occ_error_df, kappa_params, 
			depth_params, k_params, "Expected Time at State {}".format(state), state, file, False, False)

	
		# plot for single node probabilities
		for kappa in kappa_params:
			for depth in depth_params:
				file = "{}_{}_p{}_bias{}_kappa{}_depth{}_state{}.jpg".format(filename,"joint_occ",
										 p, bias, kappa, depth, state)
				plot_joint_state_prob_vs_time(joint_history_df,joint_error_history_df, kappa,depth,k_params, state, "Automata", file, False, False)

