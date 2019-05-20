#######################################################
# This simulation analyzes performance of the mean field approximation
# versus the k approximation at large times for various values
# of kappa (degree of the tree) and beta for the Ising model. 
# These two approximations are compared to MC sampled dynamics, and
# accuracy is measured by:
# the total variation distance between the joints of the root node
# and one of its neighbors, as well as the distributions of the root, 
# joint distributions with more neighbors, and trajectories as well.
#
# This comparison in particular analyzes the 1Approx.
#######################################################

import os 
import sys
sys.path.append("../SimulateProcesses/")
sys.path.append("../SimulateProcesses/Ising")
from Ising_Reimplementation import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import matplotlib

if __name__== "__main__":

	#np.random.seed(17)
	## at stationarity, which method works better?

	p = 0.9 #update parameter
	bias = -0.2
	p_0 = (1+bias)/2

	## intake what parameters to run/ analyze
	kappa_params = [2,3,4]
	beta_params = np.linspace(0,2,6)
	k = 1
	depth = 8

	# ideally want this to be larger
	T = 100

	### Collect Data
	# use Pandas, want multidimensional index; 
	# columns should be dynamics; mean field; k-approximation,various vals of k.

	columns = ["KApprox", "Mean Field", "dynamics"] 
	idx_iter = [kappa_params, beta_params]
	index = pd.MultiIndex.from_product(idx_iter, names = ['kappa','beta'])
	zero_df = np.zeros((len(kappa_params)*len(beta_params),len(columns)))

	# probability at node 0, time T 
	joint_prob_df = pd.DataFrame(zero_df,columns = columns, index = index)

	idx = pd.IndexSlice
	
	## run mean field and get desired data
	print("Running dynamics")
	for beta, kappa in itr.product(beta_params,kappa_params):
		print("{},{}".format(kappa,beta))
		(root, joint, root_occ, joint_occ, root_history, joint_history) = \
			run_dynamics(depth, kappa, T, p, beta, bias, [0], load_data = False, use_bc = False)

		joint_prob_df.loc[idx[kappa,beta],idx['dynamics']] = str(joint)

	## run mean field
	print("Running mean field")
	for beta, kappa in itr.product(beta_params,kappa_params):
		print("{},{}".format(kappa,beta))
		(root, joint, root_occ, joint_occ, root_history, joint_history) = \
			run_mean_field_simulations(kappa, p_0, T, p, beta, load_data = False)

		joint_prob_df.loc[idx[kappa,beta],idx['Mean Field']] = str(joint)


	## run local approximation
	for beta, kappa in itr.product(beta_params,kappa_params):
		print(beta, kappa)

		(root, joint, root_occ, joint_occ, root_history, joint_history) = \
			run_k_approximation(1, kappa, T, p, beta, k, bias, load_data = False) 

		joint_prob_df.loc[idx[kappa,beta],idx['KApprox']] = str(joint)

	# save the dataframes
	def get_df_string(s):
		f = "ContactProcessResults/dataframes/{}_T{}_maxkappa{}_maxbeta{}_bias{}".format(s,T,
				max(kappa_params), max(beta_params), bias)
		return f
	
	joint_prob_df.to_pickle(get_df_string("mf_local_comparison"))

	def str_to_dict(str):
		d = ast.literal_eval(str)
		return d

	# compute the distances
	total_variations_mf = np.zeros((len(kappa_params),len(beta_params)))
	total_variations_kapprox = np.zeros((len(kappa_params),len(beta_params)))
	for i in range(len(kappa_params)):
		for j in range(len(beta_params)):

			prob_mf = np.array(list(str_to_dict(joint_prob_df.loc[idx[kappa_params[i],beta_params[j]],idx['Mean Field']]).values()))
			prob_kapprox = np.array(list(str_to_dict(joint_prob_df.loc[idx[kappa_params[i],beta_params[j]],idx['Local Approx']]).values()))
			prob_dynamics = np.array(list(str_to_dict(joint_prob_df.loc[idx[kappa_params[i],beta_params[j]],idx['dynamics']]).values()))

			total_variations_mf[i,j] = np.sum(np.abs(prob_mf - prob_dynamics))
			total_variations_kapprox[i,j] = np.sum(np.abs(prob_kapprox - prob_dynamics))


	plt.gcf().clear()
	fig, ax = plt.subplots(1,1, figsize = (10,10))

	cmap = matplotlib.cm.viridis
	im1 = ax.imshow(total_variations,cmap = cmap.reversed())

	tick_size = 6.5

	ax.set_xlabel('Beta',fontsize = 10)
	ax.set_xticks(np.arange(len(beta_params)))
	ax.set_xticklabels(labels = list(beta_params),fontsize = tick_size )

	ax.set_ylabel('Kappa',fontsize = 10)
	ax.set_yticks(np.arange(len(kappa_params)))
	ax.set_yticklabels(labels = list(kappa_params),fontsize = tick_size )

	ax.set_title('Accuracy of Mean Field Approximation for Ising Model')

	divider1 = make_axes_locatable(ax)
	cax1 = divider1.append_axes('right', size='5%', pad=0.1)
	fig.colorbar(im1,cax1,orientation = 'vertical')
	plt.show()
