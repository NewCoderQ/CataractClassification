# -*- coding: utf-8 -*-
# @Author: Zhiqiang
# @Date:   2017-09-18 22:41:33
# @Last Modified by:   Zhiqiang
# @Last Modified time: 2017-09-18 23:02:57


def extract_feature(img_path):
	'''
			根据用户输入的眼底图像的名称
			来提取该图像的三个特征，并且保存在pkl文件中

		Parameters:
			img_path: 图像的文件目录

		Returns:
			None
	'''
	img_name = raw_input("Please enter the name of the image for classifying(without the ext):")
	img_path += img_name + '.jpg'
	print(img_path)





if __name__ == '__main__':
	extract_feature('../test/val/')
	


