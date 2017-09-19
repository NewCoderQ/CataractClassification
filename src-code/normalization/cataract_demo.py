# -*- coding: utf-8 -*-
# @Author: Zhiqiang
# @Date:   2017-09-18 22:41:33
# @Last Modified by:   funny_QZQ
# @Last Modified time: 2017-09-19 10:05:36

# import cv2
import sys
sys.path.append('../')
import svmutil  				# import the libsvm model for svm
import pickle
import normalFeatures
import extractColorFeature		# color
import extractGLCMFeature		# text-feature
import extractWaveFeature		# wave
import svmTrainGetWeight 		# train data process

def train_model():
	'''
		train svm model:

		Returns:
			None
	'''
	color, GLCM, wave = svmTrainGetWeight.loadData()
	svmTrainGetWeight.generateData_4(color, GLCM, wave, [0.0944924579841, 0.0702359843817, 5.20902924753])
	print('model training...')
	svmTrainGetWeight.svmTrain_4()

def generate_test_data(weight):
	'''
		generate test for image classification

		Returns:
			None
	'''
	# load data from .pkl
	print('load test data...')
	# color_feature
	color_feature = pickle.load(open('./show_data/color_feature.pkl', 'rb'))	
	# GLCM_feature
	GLCM_feature = pickle.load(open('./show_data/GLCM_feature.pkl', 'rb'))
	# wave_feature
	wave_feature = pickle.load(open('./show_data/wave_feature.pkl', 'rb'))

	# generate SVM data
	print('generate SVM data...')

	keys = color_feature.keys()
	with open('./show_data/test_data', 'wb') as test_file:
		for key in keys:
			data_str = str(100) + ' '		# 此处的100 是可以随便标注，本来应该是正确标签的位置
			# color feature
			color_value_list = list(color_feature[key])			# ndarray ——> list
			color_normal_list = normalFeatures.normalFeatures(color_value_list)		# normalization
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
			test_file.write(data_str)	

def classification():
	print('test image classification...')
	y, x = svmutil.svm_read_problem('./show_data/test_data')
	print('model loading...')
	model = svmutil.svm_load_model('./train_data/model_weight')
	p_label, p_acc, p_val = svmutil.svm_predict(y, x, model)
	for label in p_label:
		if int(label) == 0:
			print('正常 眼底图像')
		elif int(label) == 1:
			print('轻度白内障 眼底图像')
		elif int(label) == 2:
			print('中度白内障 眼底图像')
		else:
			print('重度白内障 眼底图像')

def extract_feature(img_dir):
	'''
			根据用户输入的眼底图像的名称
			来提取该图像的三个特征，并且保存在pkl文件中

		Parameters:
			img_path: 图像的文件目录

		Returns:
			None
	'''
	# 提示用户输入眼底图像的名字，不包含后缀名
	img_name = raw_input("Please enter the name of the image for classifying(without the ext):")
	img_name += '.jpg'
	# 提取测试图像的颜色特征，并且将其保存在show_data文件夹下面的.pkl文件中
	print('提取图像的颜色特征：')
	extractColorFeature.getGrayCount(img_dir, [img_name], 'show_data/')
	print('提取图像的纹理特征：')
	extractGLCMFeature.calcGLCMFeature(img_dir, [img_name], 'show_data/')
	print('提取图像的小波特征：')
	extractWaveFeature.calcWaveFeature(img_dir, [img_name], 'show_data/')
	print('图像的特征提取完成！！！')



if __name__ == '__main__':
	extract_feature('../test/val/')
	train_model()
	generate_test_data([0.0944924579841, 0.0702359843817, 5.20902924753])
	classification()
	


