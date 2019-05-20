#######################################################
# This simulation analyzes performance of the Contact Process
# for various values of the degree and depth of regular trees: the goal
# is to see when approximations for this IPS are accurate for finite trees.
#
# In particular, we compare the k approximation for various values of k,
# the mean field approximation, as well as the Monte-Carlo sampled dynamics
# of the process. (Large times can be analyzed, in particular).
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

from Contact_Reimplementation import *

if __name__== "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()

	filename = args.filename

	#np.random.seed(17)
	## at stationarity, which method works better?

	# Contact Process, below critical, above critical, at critical
	q = 0.1 #recovery rate
	p = 0.9 #infection rate
	bias = -0.8
	p_0 = (1+bias)/2

	## intake what parameters to run/ analyze
	kappa_params = [2,3,4,5,6]
	depth_params = [2,4,6]
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

	### run ordinary dynamics

	idx = pd.IndexSlice
	print("Running dynamics")
	for kappa, depth in itr.product(kappa_params,depth_params):
		print(kappa,depth)

		if kappa == 2:
			bd_condition = np.random.choice([0,1],2)
		else:
			bd_condition = np.random.choice([0,1],(kappa**depth))

		(root, joint, root_occ, joint_occ,
		  root_history, joint_history) = \
			run_dynamics(depth,kappa,T,p,q,bias,bd_condition,load_data = False, use_bc = False)

		single_node_prob_df.loc[idx[kappa,depth],idx['Dynamics']] = root
		joint_prob_df.loc[idx[kappa,depth],idx['Dynamics']] = str(joint)
		single_node_occ_df.loc[idx[kappa,depth],idx['Dynamics']] = root_occ
		joint_occ_df.loc[idx[kappa,depth],idx['Dynamics']] = str(joint_occ)

		# store the histories
		single_node_history_df.loc[:,idx['Dynamics',kappa,depth]] = root_history[:T] 
		joint_history_df.loc[:,idx['Dynamics',kappa,depth]] = [str(d) for d in joint_history][:T]


	## run mean field and get desired data
	print("Running mean field")
	for kappa in kappa_params:
		print("{}".format(kappa))
		(root, joint, root_occ, joint_occ, root_history, joint_history) = \
			run_mean_field_simulations(kappa,p_0,T,p, q, load_data = False)

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
			run_k_approximation(1, kappa, T, p, q, k, bias, load_data = False) 

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
		f = "ContactProcessResults/dataframes/{}_T{}_maxkappa{}_maxdepth{}_k{}_bias{}".format(s,T,
				max(kappa_params), max(depth_params), max(k_params), bias)
		return f
	
	single_node_prob_df.to_pickle(get_df_string("single_node_prob_df"))
	joint_prob_df.to_pickle(get_df_string("joint_prob_df"))
	single_node_occ_df.to_pickle(get_df_string("single_node_occ_df"))
	joint_occ_df.to_pickle(get_df_string("joint_occ_df"))

	single_node_history_df.to_pickle(get_df_string("single_node_history"))
	joint_history_df.to_pickle(get_df_string("joint_history_df"))
	
	######################################################
	### Plots ###
	## function of kappa

	# single node probabilities
	file = "{}_{}_q{}_p{}_bias{}.jpg".format(filename,"root_prob", q, p, bias)
	plot_vs_kappa(single_node_prob_df, kappa_params, depth_params,
		k_params, "Root Node Probability of 1", file, False, False)

	# occupation measure
	file = "{}_{}_q{}_p{}_bias{}.jpg".format(filename,"root_occ", q, p, bias)
	plot_vs_kappa(single_node_occ_df, kappa_params, depth_params,
		k_params, "Root Node, Expected Time at State 1", file, False, False, 'Expected Fraction of Time')

	## for single node results, plot as a function of T
	for kappa in kappa_params:
		for depth in depth_params:
			file = "{}_{}_q{}_p{}_bias{}_kappa{}_depth{}.jpg".format(filename,"root_prob_vs_time", q, p, bias, kappa, depth)
			plot_prob_vs_time(single_node_history_df, kappa,depth,k_params, "Contact Process", file, False, False)

	# joint probability, for a given state
	for state in [(0,0),(0,1),(1,0),(1,1)]:
		file = "{}_{}_q{}_p{}_bias{}_state{}.jpg".format(filename,"joint_prob",
										 q, p, bias,state)

		plot_joint_vs_kappa(joint_prob_df, kappa_params, 
			depth_params, k_params,"Probability of State {}".format(state), state, file, False, False)

		file = "{}_{}_q{}_p{}_bias{}_state{}.jpg".format(filename,"joint_occ",
										 q, p, bias, state)

		# joint occupation measure, for a given state
		plot_joint_vs_kappa(joint_occ_df, kappa_params, 
			depth_params, k_params, "Expected Time at State {}".format(state), state, file, False, False, 'Expected Fraction of Time')

	
		# plot for single node probabilities
		for kappa in kappa_params:
			for depth in depth_params:
				file = "{}_{}_q{}_p{}_bias{}_kappa{}_depth{}_state{}.jpg".format(filename,"joint_occ",
										 q, p, bias, kappa, depth, state)
				plot_joint_state_prob_vs_time(joint_history_df, kappa, depth, k_params, state, "Contact Process", file, False, False)
