#######################################################
# This simulation analyzes performance of the Ising Model
# for various values of the degree and depth of regular trees: the goal
# is to see when approximations for this IPS are accurate for finite trees.
#
# In particular, we compare the k approximation for various values of k,
# the mean field approximation, as well as the Monte-Carlo sampled dynamics
# of the process, and the Local approx, so that only small times can be analyzed.
#  Plots are created visualizing the following information:
#
# Probabilities of the root node
# Joint probabilities of the root and its neighbor
# Occupation measures (exp amount of time spent at a state) of the root
# ... of the root and its neighbor at joint states.
#
# This information is summarized by plotting against time or plotting
# against increasing values of kappa.
#######################################################

import os 
import sys
sys.path.append("../SimulateProcesses/")
import Create_Graphs as graphs 
import Dyadic_measures as measures
import Ising_Model as Ising 
import Monitor_Processes as monitor
import Misc
from IsingModelSimulation import *

# import standard libraries
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

import ast
import importlib

#########################################################################################################
#########################################################################################################
#########################################################################################################
### Plotting Code ###

def plot_vs_kappa(df, kappa_params, depth_params, title, scale_y = False):
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

	local_approx_vals = df.loc[idx[kappa_params,depth_params[0]],'Local Approx']
	ax.plot(kappa_params, local_approx_vals, marker = 'o', label = "Local Approx")


	plt.xlabel('Kappa')
	plt.ylabel('Probability')
	plt.xticks(kappa_params)
	if scale_y == True:
		plt.yticks(np.arange(0,1.1,0.1))
		plt.ylim(0,1)
	plt.legend()
	plt.title(title)
	plt.show()

def plot_joint_vs_kappa(df, kappa_params, depth_params, title, state, scale_y = False):

	plt.figure(figsize = (5,5))
	ax = plt.gca()

	def str_to_dict(str):
		d = ast.literal_eval(str)
		return d[state]

	idx = pd.IndexSlice

	mf_vals = df.loc[idx[kappa_params,depth_params[0]],'Mean Field'].values
	mf_vals = list(map(str_to_dict,mf_vals))
	ax.plot(kappa_params, mf_vals, marker = 'o', label = "Mean Field", linewidth = 1)


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

	local_approx_vals = df.loc[idx[kappa_params,depth_params[0]],'Local Approx'].values
	local_approx_vals = list(map(str_to_dict,local_approx_vals))
	ax.plot(kappa_params, local_approx_vals, marker = 'o', label = "Local Approx")


	plt.xlabel('Kappa')
	plt.ylabel('Probability')
	plt.xticks(kappa_params)
	if scale_y == True:
		plt.yticks(np.arange(0,1.1,0.1))
		plt.ylim(0,1)
	plt.legend()
	plt.title(title)
	plt.show()

def plot_prob_vs_time(df, kappa, depth, title, scale_y = False):
	idx = pd.IndexSlice
	plt.figure(figsize = (5,5))
	ax = plt.gca()

	
	dynamics_vals = df.loc[:,idx['Dynamics',kappa, depth]]
	errors = error_df.loc[:,idx[kappa, depth]]
	ax.errorbar(range(T), dynamics_vals,yerr = errors, label = "Dynamics")

	for k in k_params:
		kapprox_vals = df.loc[:,idx['KApprox_{}'.format(k),kappa, depth]] 
		ax.plot(range(T), kapprox_vals, label = "{}-Approximation".format(k))

	local_approx_vals = df.loc[:,idx['Local Approx',kappa,depth]]
	ax.plot(range(T), local_approx_vals, label = "Local Approx")

	mf_vals = df.loc[:,idx['Mean Field',kappa,depth]]
	ax.plot(range(T), mf_vals, label = "Mean Field")


	plt.xlabel('Time')
	plt.ylabel('Probability')
	if scale_y == True:
		plt.yticks(np.arange(0,1.1,0.1))
		plt.ylim(0,1)
	plt.legend()
	plt.title('{}, Kappa {}, Depth {}'.format(title,kappa,depth))
	plt.show()

def plot_joint_state_prob_vs_time(df, kappa, depth, state, title, scale_y = False):

	plt.figure(figsize = (5,5))
	ax = plt.gca()

	def str_to_dict(str):
		d = ast.literal_eval(str)
		return d[state]


	idx = pd.IndexSlice
	

	dynamics_vals = df.loc[:,idx['Dynamics',kappa, depth]].values
	dynamics_vals = list(map(str_to_dict,dynamics_vals))
	errors = error_df.loc[:,idx[kappa, depth]].values
	errors = list(map(str_to_dict,errors))
	ax.errorbar(range(T), dynamics_vals,yerr = errors, label = "Dynamics")

	for k in k_params:
		kapprox_vals = df.loc[:,idx['KApprox_{}'.format(k),kappa, depth]].values
		kapprox_vals = list(map(str_to_dict,kapprox_vals))
		ax.plot(range(T), kapprox_vals, label = "{}-Approximation".format(k))

	mf_vals = df.loc[:,idx['Mean Field',kappa,depth]].values
	mf_vals = list(map(str_to_dict,mf_vals))
	ax.plot(range(T), mf_vals, label = "Mean Field", linewidth = 1)

	local_approx_vals = df.loc[:,idx['Local Approx',kappa,depth]].values
	local_approx_vals = list(map(str_to_dict,local_approx_vals))
	ax.plot(range(T), local_approx_vals, label = "Local Approx")


	plt.xlabel('Time')
	plt.ylabel('Probability')
	if scale_y == True:
		plt.yticks(np.arange(0,1.1,0.1))
		plt.ylim(0,1)
	plt.legend()
	plt.title('{}, Kappa {}, Depth {},  State {}'.format(title,kappa,depth, state))
	plt.show()




if __name__== "__main__":

	## For small time, which method is better?

	# Ising Model, below critical, above critical, at critical
	beta = 0.01
	p = 0.9
	bias = -0.2
	p_0 = (1+bias)/2

	## intake what parameters to run/ analyze
	kappa_params = [2,3,4]
	depth_params = [1,2,3,4,5,6,7,8]
	k_params = [1]

	# ideally want this to be larger
	T = 4

	### Collect Data
	# use Pandas, want multidimensional index; 
	# columns should be dynamics; mean field; k-approximation,various vals of k.

	columns = ["Dynamics","Mean Field","Local Approx"] + ["KApprox_{}".format(k) for k in k_params]
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
	cols = ['Mean Field', 'Dynamics', 'Local Approx'] + ["KApprox_{}".format(k) for k in k_params]
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
	single_node_occ_error_df = pd.DataFrame(zero_df,columns = columns, index = index)
	joint_occ_error_df = pd.DataFrame(zero_df,columns = columns, index = index)

	idx = range(T)
	colnames = [kappa_params,depth_params]
	colnames = pd.MultiIndex.from_product(colnames,names = ['Kappa','Depth'])
	zero_df= np.zeros((T,len(colnames)))
	single_node_error_history_df = single_node_history_df = pd.DataFrame(zero_df, columns = colnames, index = idx)
	joint_error_history_df = single_node_history_df = pd.DataFrame(zero_df, columns = colnames, index = idx)

	idx = pd.IndexSlice
	## run ordinary dynamics
	print("Running dynamics")
	for kappa, depth in itr.product(kappa_params,depth_params):
		print(kappa,depth)
		(root, joint, root_occ, joint_occ, root_history, joint_history) = \
			run_dynamics(depth,kappa,T,p,beta,bias,load_data = False)

		single_node_prob_df.loc[idx[kappa,depth],idx['Dynamics']] = root
		joint_prob_df.loc[idx[kappa,depth],idx['Dynamics']] = str(joint)
		single_node_occ_df.loc[idx[kappa,depth],idx['Dynamics']] = root_occ
		joint_occ_df.loc[idx[kappa,depth],idx['Dynamics']] = str(joint_occ)

		# store the histories
		single_node_history_df.loc[:,idx['Dynamics',kappa,depth]] = root_history[:T] # there might be off by one error, check this out
		joint_history_df.loc[:,idx['Dynamics',kappa,depth]] = [str(d) for d in joint_history][:T]

	
	## run mean field and get desired data
	print("Running mean field")
	for kappa in kappa_params:

		(root, joint, root_occ, joint_occ, root_history, joint_history) = \
			run_mean_field_simulations(kappa,p_0,T,p, beta, load_data = False)

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
			run_k_approximation(1, kappa, T, p, beta, k, bias, load_data = True) 

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

	## run local approx
	for kappa in kappa_params:
		print("Running Local_Approx")
		print(kappa)

		(root, joint, root_occ, joint_occ, root_history, joint_history) = \
			run_local_approximation(1, kappa, T, p, beta, bias, load_data = True)

		single_node_prob_df.loc[idx[kappa,:],idx['Local Approx']] = root[1]
		joint_prob_df.loc[idx[kappa,:],idx['Local Approx']] = str(joint)
		single_node_occ_df.loc[idx[kappa,:],idx['Local Approx']] = list(root_occ.values())[1] #check this?
		joint_occ_df.loc[idx[kappa,:],idx['Local Approx']] = str(joint_occ)

		# store the histories
		h = np.array([a[1] for a in root_history[:T] ]).reshape((T,1))
		single_node_history_df.loc[:,idx['Local Approx',kappa]] =  np.broadcast_to(h,(T,len(depth_params)))

		def array_to_dict(array):
			keys = []
			vals = []
			for k,v in np.ndenumerate(array):
				keys.append(k)
				vals.append(v)
			return dict(zip(keys, vals))

		joint_h = np.array([str(array_to_dict(d)) for d in joint_history][:T]).reshape((T,1))
		joint_history_df.loc[:,idx['Local Approx',kappa]] = np.broadcast_to(joint_h,(T,len(depth_params)))


	# save the dataframes
	def get_df_string(s):
		f = "IsingModelResults/dataframes/{}_T{}_maxkappa{}_maxdepth{}_k{}_bias{}".format(s,T,
				max(kappa_params), max(depth_params), max(k_params), bias)
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
	plot_vs_kappa(single_node_prob_df, single_node_prob_error_df, kappa_params, depth_params[1:],k_params, "Root Node Probability of 1", False)

	# occupation measure
	plot_vs_kappa(single_node_occ_df, single_node_occ_error_df, kappa_params, depth_params[1:],k_params, "Root Node, Expected Time at State 1", False)

	# joint probability, for a given state
	state = (1,1)
	plot_joint_vs_kappa(joint_prob_df, joint_prob_error_df, kappa_params, 
		depth_params[1:],"Probability of State {}".format(state), state, False)

	# joint occupation measure, for a given state
	plot_joint_vs_kappa(joint_occ_df, joint_occ_error_df, kappa_params, 
		depth_params[1:],"Expected Time at State {}".format(state), state, False)

	## for single node results, plot as a function of T
	# plot for single node probabilities
	kappa = kappa_params[-1]
	depth = depth_params[-1]
	plot_prob_vs_time(single_node_history_df,single_node_error_history_df, kappa,depth,"Ising Model", False)

	plot_joint_state_prob_vs_time(joint_history_df,joint_error_history_df, kappa,depth, state, "Ising Model", False)
