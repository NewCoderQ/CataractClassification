# -*- coding: utf-8 -*-

'''
	Date:17/05/2017
	Author:NewCoderQ
	Function:normalize the wave features
'''

import numpy as np 					# calculate tools 
import pickle						# data file operation


# normalize the features
'''
	parameter:
		feature_list:a list of features
	return:
		a list of normalized features
'''
def normalFeatures(feature_list):
	# list to array
	feature_array = np.array(feature_list)
	# get the maximum
	max_val = np.max(feature_array)
	# get the minimum
	min_val = np.min(feature_array)
	# normalize
	for i in range(len(feature_list)):
		feature_list[i] = (feature_list[i] - min_val) / float(max_val - min_val)
	# print feature_list
	return feature_list


# test
if __name__ == '__main__':
	test_list = [0, 0, 23, 23, 4, 2, 0, 0]
	normalFeatures(test_list)

