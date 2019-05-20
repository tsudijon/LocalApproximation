import numpy as np 
import itertools as itr

################################################
################################################
################################################

# consider dyadic measures.
def dyadic_measure(vector,k,dimension):
	'''
	k: the fineness of the measure: the output is an nd array of side length 2^k. Models k step history
	dimension: the dimension of the nd array. Models the degree in a regular tree.
	'''

	# add some checks 

	side = 2**k
	# use numpy array to model this measure
	measure = np.array(vector, dtype = np.float32).reshape(tuple([side]*dimension))
    
	# have to normalize this measure
	return measure/measure.sum()

def get_random_dyadic_measure(k,dimension):
	return dyadic_measure(np.random.uniform(0,1,2**(dimension*k)),k,dimension)


def smoothen(measure,k):
	'''
	Input: a dyadic measure. 
	Output: a dyadic measure of side length k.
	'''
	side_len = measure.shape[0]
	dim = len(measure.shape)

	assert side_len >= 2**k

	if side_len == 2**k:
		return measure

	smoothed_measure = np.zeros(shape = tuple([2**k]*dim))
	smoothed_idxs = np.arange(0,2**k)
	smoothed_len = int(side_len/(2**k))

	for cds in itr.product(smoothed_idxs,repeat = dim):

		smaller_cube = [range(smoothed_len*cd, smoothed_len*(cd+1)) for cd in cds]
		# * is the splat operator: turns a list of arguments into inputs for an arg
		smaller_cube_cds = [list(tp) for tp in itr.product(*smaller_cube)] # n d-tuples
		coordinate_lists = list(np.array(smaller_cube_cds).T) # d lists of n coordinates
		smoothed_measure[cds] = measure[coordinate_lists].sum() #
        
	return smoothed_measure

def embed(measure,k):
	'''
	View a measure as one with greater side length
	'''
	side_len = measure.shape[0]
	dim = len(measure.shape)

	assert side_len <= 2**k

	if side_len == 2**k:
		return measure

	refined_measure = np.zeros(shape = tuple([2**k]*dim))
	refined_idxs = np.arange(0,2**k)
	refined_len = int((2**k)/side_len)

	for cds in itr.product(refined_idxs,repeat = dim):
		cds_in_old_measure = (np.array(cds)/refined_len).astype(int)
		refined_measure[cds] = measure[tuple(cds_in_old_measure)]/(refined_len**dim)

	return refined_measure


################################################
################################################
################################################
### Consider N-Ary Measures
def nary_measure(vector,k,dimension,n):
	'''
	k: the fineness of the measure: the output is an nd array of side length n^k. Models k step history
	dimension: the dimension of the nd array. Models the degree in a regular tree.
	'''

	# add some checks 

	side = n**k
	# use numpy array to model this measure
	measure = np.array(vector, dtype = np.float32).reshape(tuple([side]*dimension))
    
	# have to normalize this measure
	return measure/measure.sum()

def get_random_nary_measure(k,dimension,n):
	return dyadic_measure(np.random.uniform(0,1,n**(dimension*k)),k,dimension)


def nary_smoothen(measure,k,n):
	'''
	Input: a dyadic measure. 
	Output: a dyadic measure of side length k.
	'''
	side_len = measure.shape[0]
	dim = len(measure.shape)

	assert side_len >= n**k

	if side_len == n**k:
		return measure

	smoothed_measure = np.zeros(shape = tuple([n**k]*dim))
	smoothed_idxs = np.arange(0,n**k)
	smoothed_len = int(side_len/(n**k))

	for cds in itr.product(smoothed_idxs,repeat = dim):

		smaller_cube = [range(smoothed_len*cd, smoothed_len*(cd+1)) for cd in cds]
		# * is the splat operator: turns a list of arguments into inputs for an arg
		smaller_cube_cds = [list(tp) for tp in itr.product(*smaller_cube)] # n d-tuples
		coordinate_lists = list(np.array(smaller_cube_cds).T) # d lists of n coordinates
		smoothed_measure[cds] = measure[coordinate_lists].sum() #
        
	return smoothed_measure

def nary_embed(measure,k):
	'''
	View a measure as one with greater side length
	'''
	side_len = measure.shape[0]
	dim = len(measure.shape)

	assert side_len <= n**k

	if side_len == n**k:
		return measure

	refined_measure = np.zeros(shape = tuple([n**k]*dim))
	refined_idxs = np.arange(0,n**k)
	refined_len = int((n**k)/side_len)

	for cds in itr.product(refined_idxs,repeat = dim):
		cds_in_old_measure = (np.array(cds)/refined_len).astype(int)
		refined_measure[cds] = measure[tuple(cds_in_old_measure)]/(refined_len**dim)

	return refined_measure

















