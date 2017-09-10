# -*- coding: utf-8 -*-

'''
	做一些预处理工作
		1、获取指定目录下的所以图片的路径
			返回值为每张待提取特征图片的相对路径
		2、将原始图片进行拆分成小图片，然后对小图片进行特征提取操作
			注意：左上角那一块(大概是1/4的位置)被剔除，因为在去除病
				  人信息的时候影响了眼底照片的质量
'''

import os
import cv2
import numpy as np 
# np.set_printoptions(threshold = 'nan')

# 获片数据库中所有的图片的路径以及名称
def getImgPath():
	img_path = list()
	dir_path = './test/train/'
	dir_list = os.listdir(dir_path)
	for item in dir_list:
		img_name = os.listdir(dir_path + item + '/')
		for every_path in img_name:
			img_path.append(item + '/' + every_path)
	return img_path

# split images into small images
'''
	parameter:
		img:array of a image

	return:
		a array list of splited images 
'''
def splitImg(img):
	split_imgs = list()				# 创建一个列表，用来存放分割完成的子图像
	# img = cv2.imread(img_path, 0)
	# print type(img)
	split_imgs.append(img[:512, 512:1024])
	split_imgs.append(img[:512, 1024:1536])
	split_imgs.append(img[512:1024, 0:512])
	split_imgs.append(img[512:1024, 512:1024])
	split_imgs.append(img[512:1024, 1024:1536])
	split_imgs.append(img[512:1024, 1536:2048])
	split_imgs.append(img[1024:1536, 0:512])
	split_imgs.append(img[1024:1536, 512:1024])
	split_imgs.append(img[1024:1536, 1024:1536])
	split_imgs.append(img[1024:1536, 1536:2048])
	split_imgs.append(img[1536:2048, 512:1024])
	split_imgs.append(img[1536:2048, 1024:1536])
	return split_imgs


if __name__ == '__main__':
	splitImg(cv2.imread('test/train/0/203.jpg', 0))