"""
Remarks on the cellular automaton global
synchronisation problem – deterministic vs. stochastic
models

Nazim Fates

Implementing an idea for a stochastic CA model that converges to synchronization in finite time.
See page 15.
"""

import networkx as nx 
import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.animation as manimation
import matplotlib as mpl

def simulate_synch_ca_ring_dynamics(steps, N, p, bias):
	"""
	Defines the synchronizing problem CA on a ring.
	Params:
	---------------------
	steps: int
		number of steps to run the process for
	N: int
		number of nodes in the ring graph
	p: prob 
		param for the update making the CA synchronous
	save_file: string
		location to save figures to in order to generate video
	"""
	update_dict = {(0,0,0):1,
				   (0,0,1):1-p,
				   (0,1,0):2*p,
				   (0,1,1):p,
				   (1,0,0):1-p,
				   (1,0,1):1-2*p,
				   (1,1,0):p,
				   (1,1,1):0}

	ring = nx.cycle_graph(N)

	#initialize a random initial state
	biased_p = (1+bias)/2
	initial_state = [1 if p < biased_p else 0 for p in np.random.rand(N)]

	states = {idx:state for idx,state in enumerate(initial_state)}
	nx.set_node_attributes(ring, states, 'state')

	states = nx.get_node_attributes(ring,'state')
	history = [states]

	for s in range(steps):

		states = history[-1]

		# also asynchronous
		for node in ring.nodes:
			neighbors = [(node-1)%N, node, (node+1)%N]
			nbr_states = tuple([states[n] for n in neighbors])
			if np.random.rand() < update_dict[nbr_states]:
				ring.nodes[node]['state'] = 1 
			else:
				ring.nodes[node]['state'] = 0

		# save a copy of the graph
		history.append(nx.get_node_attributes(ring,'state'))

	"""
	pos = nx.spring_layout(ring)
	for i in range(len(history)):
		plt.clf()
		colormap = ['purple' if node == 1 else 'yellow' for node in history[i].values()]
		nx.draw(ring,node_color = colormap, pos = pos)
		plt.savefig("{}{:04d}.png".format(save_file,i))
	"""


	return history

def simulate_contrarian_voter_dynamics(graph, steps, p, bias):

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
			if np.random.rand() < contrarian_probability_of_one(current_sign, p, nbr_states):
				graph.nodes[node]['state'] = 1 
			else:
				graph.nodes[node]['state'] = 0

		history.append(nx.get_node_attributes(graph,'state'))
	"""
	pos = nx.spring_layout(graph)
	save_file = "Images/Generalization/"
	for i in range(len(history)):
		plt.clf()
		colormap = ['purple' if node == 1 else 'yellow' for node in history[i].values()]
		nx.draw(graph,node_color = colormap, pos = pos)
		plt.savefig("{}{:04d}.png".format(save_file,i))
	"""

	return history


def contrarian_probability_of_one(current_sign, p, nbr_states):
	if current_sign == 1:
		prob = (len(nbr_states) - sum(nbr_states))*p/len(nbr_states)
	else:
		prob = 1 - sum(nbr_states)*p/len(nbr_states)

	return prob


"""
python Create_video.py -dir "/Users/sianamuljadi/Desktop/synchca/" -ext "png" -o "synchronized_ca.avi"

python Create_video.py -dir "/Users/sianamuljadi/projects/'Probability Thesis'/Simulations/SimulateProcesses/Oscillator/Images/" -ext "png" -o "synchronized_ca.avi"
"""

def simulate_mean_field_dynamics(steps, kappa, p, p_0):
	"""
	Simulates the probability of a node being 1. Seems like it alternates between p_0,1-p_0
	"""
	probs = [p_0]

	for s in range(steps):
		old = probs[-1]
		new_given_0 = 1 - p*old 
		new_given_1 = p*(1-old) 

		new = new_given_0*(1-old) + new_given_1*(old)
		probs.append(new)

	return probs






