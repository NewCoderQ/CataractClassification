# -*- coding: utf-8 -*-

'''
	Date:12/05/2017
	Author:NewCoderQ
	Function:SVM train
'''

from svmutil import *			# import libsvm
import pickle					# load file
import os						# directory operation
import normalFeatures			# normalization
import time 					# analysis time

path_date = '05-17'			# define the directory


def loadData():
	# load .pkl file
	print 'data loading...'
	start_time = time.time()
	color_feature = pickle.load(open('../split_features/color_feature.pkl', 'rb'))
	GLCM_feature = pickle.load(open('../split_features/GLCM_feature.pkl', 'rb'))
	wave_feature = pickle.load(open('../split_features/wave_feature.pkl', 'rb'))
	print 'train data cost time:', int(time.time() - start_time), 's'
	return color_feature, GLCM_feature, wave_feature

# four labels
'''
	parameters:
		color_feature, GLCM_feature, wave_feature:三个特征向量
		weight:遗传算法计算的三个权重值的列表
'''
def generateData_4(color_feature, GLCM_feature, wave_feature, weight): # [4.333, 5.4, 3.4]
	color_keys = color_feature.keys()		# the keys of each feature is the same
	
	print "generate data..."
	print 'weight:', weight
	file = open(path_date + '/train_data/svm_data', 'w')
	j = 0
	for key in color_keys:
		j += 1
		# if j % 500 == 0:
		# 	print j, '/', len(color_keys)
		data_str = ""		
		data_str += key.split('/')[0] + " "
		# color feature
		color_value_list = list(color_feature[key])
		#color_value_list = [x * weight[0] for x in color_value_list]
		color_normal_list = normalFeatures.normalFeatures(color_value_list)
		# print color_normal_list
		# print '********************************'
		color_normal_list = [x * weight[0] for x in color_normal_list]
		# print color_normal_list
		# print '################################'
		i = 0							# src-data index
		for item in color_normal_list:
			i += 1
			data_str = data_str + str(i) + ":" + str(item) + " "
		# GLCM feature
		GLCM_value_list = list(GLCM_feature[key])
		#GLCM_value_list = [x * weight[1] for x in GLCM_value_list]
		GLCM_normal_list = normalFeatures.normalFeatures(GLCM_value_list)
		GLCM_normal_list = [x * weight[1] for x in GLCM_normal_list]
		for item in GLCM_normal_list:
			i += 1
			data_str = data_str + str(i) + ":" + str(item) + " "
		# wave feature
		wave_value_list = list(wave_feature[key])
		#wave_value_list = [x * weight[2] for x in wave_value_list]
		wave_normal_list = normalFeatures.normalFeatures(wave_value_list)
		wave_normal_list = [x * weight[2] for x in wave_normal_list]
		for item in wave_normal_list:
			i += 1
			data_str = data_str + str(i) + ":" + str(item) + " "

		data_str += '\n'
		file.write(data_str)


		
		
# svm train_4
def svmTrain_4():
	# y, x = svm_read_problem(path_date + '/train_data/svm_rawed_data')
	y, x = svm_read_problem(path_date + '/train_data/svm_data')
	model = svm_train(y, x, '-c 32 -g 0.0078125')		# train model
	
	# accuracy_train = svm_train(y, x, '-c 32 -g 0.0078125 -v 5')		# train model
	# print '###########################################'
	# print 'accuracy_train:', accuracy_train
	# print '###########################################'
	# return accuracy_train
	svm_save_model(path_date + '/train_data/model_weight', model)


if __name__ == '__main__':
	# multiSVM()
	# svmImplement()
	# svmTrain()
	color_feature, GLCM_feature, wave_feature = loadData()
	generateData_4(color_feature, GLCM_feature, wave_feature)
	# normalize the train_data
	print 'normalize the train_data...'
	cmd = 'cd 05-17/train_data&svm-scale -l 0 -u 1 -s raw_data svm_data > svm_rawed_data'
	os.system(cmd)
	print 'normalization done!'
	print 'data training...'
	svmTrain_4()
	print 'model train done!'
	

