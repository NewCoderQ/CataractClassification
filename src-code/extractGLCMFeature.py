# -*- coding: utf-8 -*-

'''
	计算图像的灰度共生矩阵
		1、将图像的像素值范围缩减到16位
		2、分别计算水平，垂直，45度，135度的灰度共生矩阵
	根据灰度共生矩阵计算特征
		选取的特征有：能量，对比度，逆差矩和熵
'''

import numpy as np
# np.set_printoptions(threshold='nan')
import cv2
import math
import preProcess						# 获取需要进行特征提取的图片的相对路径
import pickle


# 对图像的灰度值进行降维，为了简化之后的灰度共生矩阵的计算	23.4s
def decDim(src_array):									# 引用传值
	print "decDim..."
	for i in range(src_array.shape[0]):					# 遍历行数
		# print 'the %dth line' % i
		for j in range(src_array.shape[1]):				# 遍历列数
			src_array[i][j] = src_array[i][j] / 16


#***************************calculate GLCM***************************
# 结果矩阵的初始化，用来存储灰度共生矩阵
def initMat():											# 生成一个16 * 16的全零矩阵
	dst_mat = np.zeros((16, 16), dtype = np.int)
	# dst_mat = np.zeros((3, 3), dtype = np.int)
	return dst_mat

# 计算水平方向上的灰度共生矩阵
def calcHorisonGLCM(img):
	print 'calculate horison GLAM...'
	dst_mat = initMat()									# 初始化灰度共生矩阵
	height, width = img.shape							# 获取输入图像的尺寸
	# 遍历图像中每个像素点的像素值
	for i in range(height):								# 行遍历
		for j in range(width - 1):						# 列遍历(此处注意下标越界错误)
			# print 'the %dth row, the %dth col' % (i, j)
			row_pixel = img[i][j]						# 获取原图像的像素值 
			col_pixel = img[i][j + 1]
			dst_mat[row_pixel][col_pixel] += 1			# 根据像素值在灰度共生矩阵上进行标注
	return dst_mat

# 计算垂直方向上的灰度共生矩阵
def calcVerticalGLCM(img):
	print 'calculate vertical GLAM...'
	dst_mat = initMat()									# 初始化灰度共生矩阵
	height, width = img.shape
	for i in range(height - 1):							# 此处注意下标越界错误
		for j in range(width):
			row_pixel = img[i][j]
			col_pixel = img[i + 1][j]					# 下一行的像素值
			dst_mat[row_pixel][col_pixel] += 1			# 标注在灰度共生矩阵上
	return dst_mat

# 计算45度方向上的灰度共生矩阵
def calc45GLCM(img):
	print 'calculate 45 GLCM...'
	dst_mat = initMat()									# 初始化灰度共生矩阵
	height, width = img.shape							# 获得图像像素矩阵的维度
	for i in range(height - 1):
		for j in range(width - 1):
			row_pixel = img[i][j]
			col_pixel = img[i + 1][j + 1]
			dst_mat[row_pixel][col_pixel] += 1
	return dst_mat

# 计算135度方向上的灰度共生矩阵
def calc135GLCM(img):
	print 'calculete 135 GLCM...'
	dst_mat = initMat()									# 初始化灰度共生矩阵
	height, width = img.shape
	for i in range(height - 1):
		for j in range(1, width):
			row_pixel = img[i][j]
			col_pixel = img[i + 1][j - 1]
			dst_mat[row_pixel][col_pixel] += 1
	return dst_mat


#****************************calculate GLCM features****************************
# 对灰度共生矩阵进行归一化处理
def normalGLCM(src_mat):
	# 计算灰度共生矩阵中所有元素的和
	height, width = src_mat.shape						# 获取输入矩阵的规格
	sum = 0												# 定义一个变量，用来存储矩阵中所有元素的和
	for i in range(height):
		for j in range(width):
			sum += src_mat[i][j]						# 求和
	# 将矩阵中的每个元素 / 元素之和
	dst_mat = np.multiply(src_mat, (1.0 / sum))
	return dst_mat

# 对灰度共生矩阵进行相关的特征计算
# 特征包括：对比度(反差)、相关度、能量和熵
def calcFeatures(src_mat):
	src_mat = normalGLCM(src_mat)						# 对输入的灰度共生矩阵进行归一化操作
	# print "normal_img:\n", src_mat
	height, width = src_mat.shape						# 获取矩阵的尺寸
	# 变量的定义与初始化
	GLCM_energy = 0.0; GLCM_contrast = 0.0; GLCM_idMoment = 0.0; GLCM_entropy = 0.0
	for i in range(height):								# 行遍历
		for j in range(width):							# 列遍历
			GLCM_energy += np.square(src_mat[i][j])		# 灰度共生矩阵的能量计算
			GLCM_contrast += np.square(i - j) * src_mat[i][j]		# 灰度共生矩阵的对比度计算
			GLCM_idMoment += src_mat[i][j] / (1 + np.square(i - j))	# 灰度共生矩阵的逆差矩计算

			# 计算灰度共生矩阵熵的时候，因为式子中需要计算log，所有需要对矩阵中元素是否大于0进行判断
			if (src_mat[i][j] > 0):
				GLCM_entropy -= src_mat[i][j] * math.log(src_mat[i][j])	# 灰度共生矩阵的熵的计算
	# 将灰度共生矩阵特征计算结果:能量，对比度，逆差矩和熵， 存入到一个列表中
	GLCM_features = list()
	GLCM_features.append(GLCM_energy)
	GLCM_features.append(GLCM_contrast)
	GLCM_features.append(GLCM_idMoment)
	GLCM_features.append(GLCM_entropy)
	return GLCM_features								# 将特征列表返回

# 编写一个函数，方便其他的模块进行调用
'''
	parameter:
		img_dir:图片文件的上级目录
		img_path:图片文件的具体名称
		fea_dir:图片特征的保存路径的上级目录
'''
def calcGLCMFeature(img_dir, img_paths, fea_dir):
	GLCM_feature = {}									# 定义一个字典用来存储每张图片的灰度共生矩阵的特征
														# 键为图片的名称，值为四个方向的灰度共生矩阵的特征列表
														# 灰度共生矩阵的特征列表是一个16维的向量
	# img_path = "./test/train/0/1.jpg"
	for i in range(len(img_paths)):
		# print 'processing %d / %d image' % ((i + 1), len(img_paths))
		img_path = img_paths[i]
		src_img = cv2.imread(img_dir + img_path, 0)		# 将图像以灰度图的形式读入
		
		split_feature_list = list()						# 用来存放分割图片的灰度共生矩阵特征
		split_img_list = preProcess.splitImg(src_img)
		for j in range(len(split_img_list)):
			if i % 50 == 0:
				print 'processing %d / %d / %d image' % ((i + 1), len(img_paths), (j + 1))
			# 将原图像的尺寸缩小，由2048 * 2048变为512 * 512，缩短提取特征的时间
			img = cv2.resize(split_img_list[j], (128, 128), interpolation = cv2.INTER_CUBIC)
			print "the size of dst_img is: ", img.shape

			#*****************************************
			# only for test
			# src_mat = np.mat([[0, 1, 2, 0, 1, 2], 
			# 				  [1, 2, 0, 1, 2, 0], 
			# 				  [2, 0, 1, 2, 0, 1], 
			# 				  [0, 1, 2, 0, 1, 2], 
			# 				  [1, 2, 0, 1, 2, 0], 
			# 				  [2, 0, 1, 2, 0, 1]]).A
			#*****************************************
			feature = []										# 定义一个列表用来存储每张图片的灰度共生矩阵的特征
																# 该列表为一个16维的特征向量
			decDim(img)											# 对原图像的灰度值进行降维操作
			
			# 计算灰度共生矩阵
			GLCM_h = calcHorisonGLCM(img)								# 计算水平共生矩阵
			# print "GLCM_h:\n", GLCM_h
			GLCM_v = calcVerticalGLCM(img)								# 计算垂直共生矩阵
			GLCM_45 = calc45GLCM(img)									# 计算45度共生矩阵
			GLCM_135 = calc135GLCM(img)									# 计算135度共生矩阵

			# 每个方向的灰度共生矩阵的四个特征的计算，返回值为一个4维的特征向量
			feature.extend(calcFeatures(GLCM_h))								# 水平方向			
			# print "feature_h:\n", feature
			feature.extend(calcFeatures(GLCM_v))								# 垂直方向
			feature.extend(calcFeatures(GLCM_45))								# 45度方向
			feature.extend(calcFeatures(GLCM_135))								# 135度方向
			# print "feature_list:\n", feature
			split_feature_list.extend(feature)
		GLCM_feature[img_path.split('.')[0]] = split_feature_list 						# 将图像的灰度共生矩阵特征保存在字典中
		print len(GLCM_feature)
	# 最后，将提取的灰度共生矩阵的特征字典存储在.pkl文件中
	pickle.dump(GLCM_feature, open(fea_dir + 'GLCM_feature.pkl', 'wb'))


# 测试函数，用来提取训练数据集的灰度共生矩阵的特征
if __name__ == '__main__':
	# print getImgPath()[0]
	img_path_list = preProcess.getImgPath()
	calcGLCMFeature('./test/train/', img_path_list, './split_features/')
	
