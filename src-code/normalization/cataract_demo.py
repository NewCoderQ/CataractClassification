# -*- coding: utf-8 -*-
# @Author: Zhiqiang
# @Date:   2017-09-18 22:41:33
# @Last Modified by:   funny_QZQ
# @Last Modified time: 2017-09-19 00:14:35

# import cv2
import sys
sys.path.append('../')
import svmutil  				# import the libsvm model for svm
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
	svmTrainGetWeight.generateData_4(color, GLCM, wave, [0.13637379083,	0.299172949594,	6.65003458262])
	print('model training...')
	svmTrainGetWeight.svmTrain_4()


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
	


