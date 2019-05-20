#######################################################
# This simulation analyzes performance of the mean field approximation
# versus the k approximation for at large times for various values
# of kappa (degree of the tree) and beta for the Ising model. 
# These two approximations are compared to MC sampled dynamics, and
# accuracy is measured by:
# the total variation distance between the joints of the root node
# and one of its neighbors, as well as the distributions of the root, 
# joint distributions with more neighbors, and trajectories as well.
#
# This comparison in particular analyzes the 2Approx near the critical regime.
#######################################################

import os 
import sys
sys.path.append("../SimulateProcesses/")
sys.path.append("../SimulateProcesses/Ising")
from mpl_toolkits.axes_grid1 import make_axes_locatable

import IsingMonitor as monitor

import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

if __name__== "__main__":

	#np.random.seed(17)
	## at stationarity, which method works better?

	p = 0.9 #update parameter
	bias = -0.2
	p_0 = (1+bias)/2

	## intake what parameters to run/ analyze

	kappa_params = [2,3,4,5,6,7,8]
	beta_params = np.array(np.linspace(0,2,11))
	kb_params = np.array([[6,0.2],
						 [6,0.4],
						 [5,0.4],
						 [4,0.4],
						 [4, 0.6000000000000001],
						 [4,0.8],
						 [3, 0.6000000000000001],
						 [3,0.8],
						 [3,1.0],
						 [3,1.2000000000000002],
						 [3,1.4000000000000001],
						 [3,1.6],
						 [3,1.8],
						 [3,2.0]])
	k = 2
	depth = 8

	# ideally want this to be larger
	T = 50

	### Collect Data
	# use Pandas, want multidimensional index; 
	# columns should be dynamics; mean field; k-approximation,various vals of k.

	# save the dataframes
	def get_df_string(s):
		f = "IsingModelResults/{}_T{}_maxkappa{}_maxbeta{}_bias{}".format(s,T,
				8, 2.0, -0.2)
		return f
	
	joint_prob_df = pd.read_pickle(get_df_string("mf_local_comparison"))

	def str_to_dict(str):
		d = ast.literal_eval(str)
		return d

	# compute the distances
	total_variations_kapprox = np.zeros((len(kappa_params),len(beta_params)))
	for i in range(len(kb_params)):

		# calculate dynamics
		idx = pd.IndexSlice
		prob_dynamics = np.array(list(str_to_dict(joint_prob_df.loc[idx[int(kb_params[i,0]),kb_params[i,1]],idx['dynamics']]).values()))

		# get the kapprox info
		filename = "IsingModelResults/kapprox/kapprox_kappa{}_p0.9_beta{:.1f}_T{}_k{}_bias{}.npy".format(int(kb_params[i,0]),kb_params[i,1], T, k, bias)

		measure_history = np.load(filename)

		process = monitor.MonitorLocalApproximation(np.zeros((2,2,2)),p,kb_params[i,1])
		process.measure_history = measure_history


		joint2_la = process.get_fixed_time_measure((0,1),T-1)
		prob_kapprox = joint2_la.reshape(1,4)

		total_variations_kapprox[int(kb_params[i,0]) - 2,int(kb_params[i,1]/0.2)] = np.sum(np.abs(prob_kapprox - prob_dynamics))


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

	visualize_total_var(total_variations_kapprox, "Accuracy of 2-Approximation at Critical Region")
