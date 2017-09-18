# -*- coding: utf-8 -*-
# @Author: Zhiqiang
# @Date:   2017-09-18 22:41:33
# @Last Modified by:   Zhiqiang
# @Last Modified time: 2017-09-18 23:07:34

import cv2

def extract_feature(img_path):
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
	img_path += img_name + '.jpg'
	test_img = cv2.imread(img_name)
	print(test_img.shape)







if __name__ == '__main__':
	extract_feature('../test/val/')
	


