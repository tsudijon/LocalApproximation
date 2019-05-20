from Contact_Reimplementation import *



	# Contact Process, below critical, above critical, at critical
	q = 0.05 #recovery rate
	p = 0.9 #infection rate
	bias = -0.8
	p_0 = (1+bias)/2

	## intake what parameters to run/ analyze
	kappa_params = [2]
	depth_params = [10,20,40,80]
	k_params = [1]

	# ideally want this to be larger
	T = 40

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


	df = single_node_history_df
	idx = pd.IndexSlice
	plt.figure(figsize = (5,5))
	ax = plt.gca()

	T = len(df) - 2

	mf_vals = df.loc[:T-1,idx['Mean Field',kappa,depth]]
	ax.plot(range(T), mf_vals, label = "Mean Field")


	for depth in depth_params:
		dynamics_vals = df.loc[:T-1,idx['Dynamics',kappa, depth]]
		ax.plot(range(T), 1 - dynamics_vals, label = "Complete Graph N = {}".format(depth))

	plt.xlabel('Time')
	plt.ylabel('Probability Root Node is 1')
	plt.yticks(np.arange(0,1.1,0.1))
	plt.ylim(0,1)
	plt.legend()
	plt.title('Mean Field Approximation on Complete Graphs')
	plt.show()
