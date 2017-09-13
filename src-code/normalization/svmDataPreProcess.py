# -*- coding: utf-8 -*-

'''
	Date:12/05/2017
	Author:NewCoderQ
	Function:pre-process the feature data for SVM
'''

import pickle			# file operation
import normalFeatures	# normalization
import os				# system operation
import time

path_date = '05-17'

# load test data
def loadData():
	start_time = time.time()
	# load .pkl file
	print 'data loading...'
	# color_feature
	color_feature = pickle.load(open('../split_predict_features/color_feature.pkl', 'rb'))	
	# GLCM_feature
	GLCM_feature = pickle.load(open('../split_predict_features/GLCM_feature.pkl', 'rb'))
	# wave_feature
	wave_feature = pickle.load(open('../split_predict_features/wave_feature.pkl', 'rb'))
	print 'test data cost time:', time.time() - start_time, 's'
	return color_feature, GLCM_feature, wave_feature


# generate data from feature_dict
'''
	return:
		a list of image-name
'''
def generateData(color_feature, GLCM_feature, wave_feature, weight):
	# match image name with label
	img_label = matchImgLabel()
	print "generate SVM feature data..."
	file = open(path_date + '/test_data/test_data', 'w')
	keys = color_feature.keys()		# image name
	j = 0
	for key in keys:
		j += 1
		# data_str = '100 '			# 注意：这个值是随便设的，但是不能没有，这是libsvm输入数据格式要求的
									# 数字后面还有一个空格，不能丢掉，不然会报错
		# if (j % 500 == 0):
		# 	print j, '/', len(keys)
		data_str = str(img_label[key]) + ' '
		# color feature
		color_value_list = list(color_feature[key])			# ndarray ——> list
		color_normal_list = normalFeatures.normalFeatures(color_value_list)
		color_normal_list = [x * weight[0] for x in color_normal_list]			# 加权
		i = 0							# src-data index
		for item in color_normal_list:
			i += 1
			data_str = data_str + str(i) + ":" + str(item) + " "
		# GLCM feature
		GLCM_value_list = list(GLCM_feature[key])
		GLCM_normal_list = normalFeatures.normalFeatures(GLCM_value_list)
		GLCM_normal_list = [x * weight[1] for x in GLCM_normal_list]
		for item in GLCM_normal_list:
			i += 1
			data_str = data_str + str(i) + ":" + str(item) + " "
		# wave feature
		wave_value_list = list(wave_feature[key])
		wave_normal_list = normalFeatures.normalFeatures(wave_value_list)
		wave_normal_list = [x * weight[2] for x in wave_normal_list]
		for item in wave_normal_list:
			i += 1
			data_str = data_str + str(i) + ":" + str(item) + " "

		data_str += '\n'
		file.write(data_str)	
	return keys					# 用于接下来找到正确的对应的标签

# match image name with label
def matchImgLabel():
	# load img_name_label.pkl
	return pickle.load(open('img_name_label.pkl', 'rb'))


if __name__ == '__main__':
	color, GLCM, wave = loadData()
	img_name = generateData(color, GLCM, wave)
	# print 'normalize the train_data...'
	# cmd = 'cd 05-17/test_data&svm-scale -l 0 -u 1 -s raw_data test_data > test_rawed_data'
	# os.system(cmd)
	# print 'normalization done!'
