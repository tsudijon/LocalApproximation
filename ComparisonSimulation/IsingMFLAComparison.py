#######################################################
# This simulation analyzes performance of the mean field approximation
# versus the local approximation for small times, for various values
# of kappa (degree of the tree) and beta for the Ising model. 
# These two approximations are compared directly, and measured by
# the total variation distance between the joints of the root node
# and one of its neighbors, as well as the distributions of the root, 
# joint distributions with more neighbors, and trajectories as well.
#
# There is plotting code to visualize the total variation distances.
#######################################################

import os 
import sys
sys.path.append("../SimulateProcesses")
sys.path.append("../SimulateProcesses/Ising")
from IsingModelSimulation import run_mean_field_simulations
import Create_Graphs as graphs 
import Dyadic_measures as measures
import Ising_Model as Ising 
import IsingMonitor as monitor
import Misc

# import standard libraries
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 
import itertools as itr
import functools
import numpy as np 

import ast
import importlib
import argparse 

from mpl_toolkits.axes_grid1 import make_axes_locatable

p = 0.9 #update parameter
bias = -0.2
p_0 = (1+bias)/2

## intake what parameters to run/ analyze
kappa_params = [2,3,4,5,6,7]
beta_params = np.array(np.linspace(0,2,11))
# k = [1]

# ideally want this to be larger
T = 3

### Collect Data
# use Pandas, want multidimensional index; 
# columns should be dynamics; mean field; k-approximation,various vals of k.

columns = ["Local Approx", "Mean Field"] 
idx_iter = [kappa_params, beta_params]
index = pd.MultiIndex.from_product(idx_iter, names = ['kappa','beta'])
zero_df = np.zeros((len(kappa_params)*len(beta_params),len(columns)))

# probability at node 0, time T 
joint2_prob_df = pd.DataFrame(zero_df,columns = columns, index = index)

def create_product_measure(bernoulli_p, dim):
	initial = np.zeros(tuple([2]*(dim)))
	for idx in itr.product([0,1],repeat = dim):
		initial[idx] = ((bernoulli_p)**(sum(idx)))*((1-bernoulli_p)**(dim + 1 - sum(idx)))

	return measures.dyadic_measure(initial,k = 1, dimension = dim)

def create_product_measure_from_vec(vec, dim):
	probs = np.array([1 - np.array(vec),vec])
	initial = np.zeros(tuple([len(vec)]*(dim)))
	for idx in itr.product(list(range(len(vec))),repeat = dim):
		initial[idx] = np.product([vec[i] for i in idx])

	return initial

def idx2bin(idx):
	return int(np.dot(np.power(2,range(len(idx))),idx))


# compute the distances
total_variations_2_marginal = np.zeros((len(kappa_params),len(beta_params)))
total_variations_root_marginal = np.zeros((len(kappa_params),len(beta_params)))
total_variations_3_marginal = np.zeros((len(kappa_params),len(beta_params)))
total_variations_kappa_marginal = np.zeros((len(kappa_params),len(beta_params)))
total_variations_history = np.zeros((len(kappa_params),len(beta_params)))


idx = pd.IndexSlice
## run local approximation
for i in range(len(kappa_params)):
	for j in range(len(beta_params)):

		print(kappa_params[i],beta_params[j])
	### Get the mean field data
		(root, joint, root_occ, joint_occ, root_history, joint_history) = \
			run_mean_field_simulations(kappa_params[i], p_0, T, p, beta_params[j], load_data = False)

		single_mf = np.array([1-root, root])
		joint2_mf= create_product_measure(root,2)
		joint3_mf = create_product_measure(root,3)
		joint_kappa_mf = create_product_measure(root,kappa_params[i]+1)

		# to get joint history mf, encode root history via binary
		probs = np.array([1-np.array(root_history),root_history])
		vec = [0]*(2**T)
		for idx in itr.product([0,1], repeat = T):
			vec[idx2bin(tuple(reversed(idx)))] = np.product(probs[idx,range(T)])

		joint_history_mf = create_product_measure_from_vec(vec, kappa_params[i] + 1)

		### get the local approximation data
		filename = "/Users/timothysudijono/Desktop/localapproxdata/localapprox_kappa{}_p0.9_beta{:.1f}_T{}_bias{}.npy".format(kappa_params[i],beta_params[j], T, bias)
		measure_history = np.load(filename)

		process = monitor.MonitorLocalApproximation(np.zeros((2,2,2)),p,beta_params[j])
		process.measure_history = measure_history

		single_la = process.get_fixed_time_measure([0],T-1)
		joint2_la = process.get_fixed_time_measure((0,1),T-1)
		joint3_la = process.get_fixed_time_measure((0,1,2),T-1)
		joint_kappa_la = process.get_fixed_time_measure(list(range(kappa_params[i] + 1)),T-1)

		joint_history_la = measure_history[-1]

		total_variations_root_marginal[i,j] = np.sum(np.abs(single_la - single_mf))
		total_variations_2_marginal[i,j] = np.sum(np.abs(joint2_la - joint2_mf))
		total_variations_3_marginal[i,j] = np.sum(np.abs(joint3_la - joint3_mf))
		total_variations_kappa_marginal[i,j] = np.sum(np.abs(joint_kappa_la - joint_kappa_mf))
		total_variations_history[i,j] = np.sum(np.abs(joint_history_la - joint_history_mf))

def visualize_total_var(array, title):
	fig, ax = plt.subplots(1,1, figsize = (10,10))

	cmap = matplotlib.cm.viridis
	norm = plt.Normalize(0, 2)

	betas = np.array(np.linspace(0,2,81))
	kappas = 2*np.exp(2*betas)/(np.exp(2* betas) - 1)
	ax.plot(5.5*betas-0.5, kappas - 2, color = 'red', label = "Critical Value")
	im1 = ax.imshow(array[::-1,],cmap = cmap.reversed(), norm = norm, extent = [0 - 0.5,len(beta_params)-1 + 0.5,0 - 0.5,len(kappa_params)-1 + 0.5])

	tick_size = 6.5

	ax.set_xlabel('Beta',fontsize = 10)
	ax.set_xticks(np.arange(len(beta_params)))
	ax.set_xticklabels(labels = ["{:0.1f}".format(beta) for beta in beta_params],fontsize = tick_size )

	ax.set_ylabel('Kappa',fontsize = 10)
	ax.set_yticks(np.arange(len(kappa_params)))
	ax.set_yticklabels(labels = list(kappa_params),fontsize = tick_size )
	#ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.0))

	ax.set_title(title)

	for i in range(len(kappa_params)):
		for j in range(len(beta_params)):
			text = ax.text(j, i, "{:0.2f}".format(array[i, j]),
				ha="center", va="center", color="w")



	divider1 = make_axes_locatable(ax)
	cax1 = divider1.append_axes('right', size='5%', pad=0.1)
	fig.colorbar(im1,cax1,orientation = 'vertical')

	plt.show()

visualize_total_var(total_variations_root_marginal, "Total Variation of Root")
visualize_total_var(total_variations_2_marginal, "Total Variation of 2 Node Joint")
visualize_total_var(total_variations_3_marginal, "Total Variation of 3 Node Joint")
visualize_total_var(total_variations_kappa_marginal, "Total Variation of Kappa Node Joint")
visualize_total_var(total_variations_history, "Total Variation of Kappa Node Trajectory")






