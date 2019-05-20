import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_regular_tree(kappa,depth):
	'''
	kappa: integer representing the degree of the nodes
	depth: integer representing the depth of the tree
	'''
	g = nx.balanced_tree(r = kappa - 1, h = depth)
	subtree = nx.balanced_tree(r = kappa - 1, h = depth - 1)

	full_tree = nx.disjoint_union(g,subtree)
	subtree_root = len(g)
	# need to append another branch onto this tree
	full_tree.add_edge(0,subtree_root)

	return full_tree

def complete_graph(n):
	return nx.complete_graph(n)

def create_ring(n):
	return nx.cycle_graph(n)

def create_lattice(depth,dim):
	"""
	Creates 2*depth^dim lattice grid.
	"""
	return nx.grid_graph(dim = [range(-depth,depth)]*dim)

def get_erdos_renyi_graph(n,p):
	return nx.erdos_renyi_graph(n,p)

def get_preferential_attachment_graph(num_nodes, out_degree):
	return nx.barabasi_albert_graph(num_nodes, out_degree)

def create_galton_watson_tree(child_dist,generations):
	"""
	Simulate a galton watson tree, where child dist is the probabilities of having 
	a specified number of offspring
	"""
	tree = nx.Graph()
	tree.add_node(0)

	current_max = 1
	child_q = [0]
	new_child_q = []
	
	for d in range(generations):
		if len(child_q) == 0:
			break
		while len(child_q) != 0:
			node = child_q.pop(0)
			num_children = vector_sample(child_dist,1)[0]
			for i in range(current_max, current_max + num_children):
				tree.add_node(i)
				tree.add_edge(node,i)
				new_child_q.append(i)

			current_max += num_children

		child_q = new_child_q
		new_child_q = []


	return tree 

def vector_sample(ps, num_samples):
	bins = np.cumsum(ps)
	return list(np.digitize(np.random.rand(num_samples), bins, right = False))

