# -*- coding: utf-8 -*-
# @Author: Zhiqiang
# @Date:   2017-09-18 22:41:33
# @Last Modified by:   funny_QZQ
# @Last Modified time: 2017-09-18 23:48:05

# import cv2
import sys
sys.path.append('../')
import extractColorFeature		# color
import extractGLCMFeature		# text-feature
import extractWaveFeature		# wave

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
	extractColorFeature.getGrayCount(img_dir, [img_name], 'show_data/')
	extractGLCMFeature.calcGLCMFeature(img_dir, [img_name], 'show_data/')
	extractWaveFeature.calcWaveFeature(img_dir, [img_name], 'show_data/')










if __name__ == '__main__':
	extract_feature('../test/val/')
	


