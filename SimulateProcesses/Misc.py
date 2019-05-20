import numpy as np

def sum_dicts(d1,d2):
	dict_sum = dict()
	for key in d1.keys():
		dict_sum[key] = d1[key] + d2[key]
	return dict_sum

def convert_to_zero_one(dict_list):
	return [ {key:int(0.5*(value + 1)) for key,value in d.items()} for d in dict_list] # what the fuck?

def append_dicts(d1,d2):
	"""
	Need to have the same keys.
	"""
	for key in d1.keys():
		(d1[key]).append(d2[key])
	return d1

