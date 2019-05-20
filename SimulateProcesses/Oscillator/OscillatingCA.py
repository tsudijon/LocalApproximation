import networkx as nx 
import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.animation as manimation
import matplotlib as mpl

def simulate_stochastic_frequency_dynamics(graph, steps, p, bias):

	#initialize a random initial state
	biased_p = (1+bias)/2
	initial_state = [1 if p < biased_p else 0 for p in np.random.rand(len(graph.nodes))]

	states = {idx:state for idx,state in enumerate(initial_state)}
	nx.set_node_attributes(graph, states, 'state')

	states = nx.get_node_attributes(graph,'state')
	history = [states]

	for s in range(steps):

		states = history[-1]
		# also asynchronous
		for node in graph.nodes:

			nbr_states = [states[n] for n in graph.neighbors(node)]
			current_sign = graph.nodes[node]['state']
			if np.random.rand() < stochastic_frequency_probability_of_one(current_sign, p, nbr_states):
				graph.nodes[node]['state'] = 1 
			else:
				graph.nodes[node]['state'] = 0

		history.append(nx.get_node_attributes(graph,'state'))

	# pos = nx.spring_layout(graph)
	# save_file = "Images/StochFreq/"
	# for i in range(len(history)):
	# 	plt.clf()
	# 	colormap = ['purple' if node == 1 else 'yellow' for node in history[i].values()]
	# 	nx.draw(graph,node_color = colormap, pos = pos)
	# 	plt.savefig("{}{:04d}.png".format(save_file,i))

	return history

def stochastic_frequency_probability_of_flip(current_sign, p, nbr_states):
	q = (1-p)/2
	prob = q + p*sum(np.array(nbr_states)== current_sign)/len(nbr_states)

	return prob

def stochastic_frequency_probability_of_one(current_sign, p, nbr_states):
	q = (1-p)/2
	prob = q + p*sum(np.array(nbr_states) == current_sign)/len(nbr_states)

	if current_sign == 0:
		return prob
	else:
		return 1 - prob

def simulate_mean_field_dynamics(T,kappa,p,p_0):

	ps = [p_0]
	q = (1-p)/2

	for i in range(T-1):
		old_p = ps[-1]

		p_given_0 = q + p*old_p
		p_given_1 = q + p*(old_p)

		new_p = p_given_1*old_p + p_given_0*(1-old_p)

		ps.append(new_p)

	return ps


