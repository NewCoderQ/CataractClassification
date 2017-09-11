# -*- coding: utf-8 -*-

'''
	Date:06/05/2017
	Author:NewCoderQ
	提取图像的纹理特征
		1、利用小波变换，将图像由时域变换到频域
		2、在频域上抽取图像的多层变换的系数特征
'''

import pywt									# 小波变换的python库
import numpy as np
import cv2
import preProcess							# 读取文件的一些基本的操作
import pickle								# 数据保存与重构


# 提取图像的三层小波分解系数
def getWaveCoe(img):						# 参数：图像
	feature_list = []						# 创建一个列表，用来存放小波系数频数
	img_wave_result = pywt.wavedec2(img, 'haar', level = 3)		
	# 对灰度图像进行小波三层小波分解
	# 选用haar小波基
	# 返回值为一个list，每一层的高频都是包含在一个tuple中
	# 例如三层的话返回为 [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2)， (cH1, cV1, cD1)]
	# 其中cA3为第三层小波分解的低频系数

	# 对三层小波分解后的系数进行分析
	l3 = img_wave_result[0]							# 第三次小波分解的低频部分
	h3 = img_wave_result[1]							# 第三次小波分解的高频部分，为一个tuple
	h3_h = h3[0]; h3_v = h3[1]; h3_d = h3[2]		# 将高频部分分别拆分为水平，垂直以及对角线三个方向
	h2 = img_wave_result[2]							# 第二次小波分解的高频部分，tuple
	h2_h = h2[0]; h2_v = h2[1]; h2_d = h2[2]
	# 由于维数较大，计算比较复杂，被舍弃
	# h1 = img_wave_result[3]						# 第一次小波分解的高频部分，tuple
	# h1_h = h1[0]; h1_v = h1[1]; h1_d = h1[2]
	
	# print "calculate the 3rd horizon..."
	feature_list.extend(getWaveFreq(h3_h))
	# print "calculate the 3rd vertical..."
	feature_list.extend(getWaveFreq(h3_v))
	# print "calculate the 3rd diagonal..."
	feature_list.extend(getWaveFreq(h3_d))
	# print "calculate the 2nd horizon..."
	feature_list.extend(getWaveFreq(h2_h))
	# print "calculate the 2nd vertical..."
	feature_list.extend(getWaveFreq(h2_v))
	# print "calculate the 2nd diagonal..."
	feature_list.extend(getWaveFreq(h2_d))
	# print 'ndim:', len(feature_list)
	return feature_list 							# 将整合完成的小波系数频数列表返回
	

# 对小波系数进行归一化处理，划分成十个等级-450 ~ 450
def getWaveFreq(wave_mat):
	wave_freq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]		# 定义一个1 * 10的全零list
	height, width = wave_mat.shape					# 获取输入矩阵的行数和列数

	# 根据指定的范围对小波系数进行归一化处理
	for i in range(height):							# 行遍历
		for j in range(width):						# 列遍历
			if(wave_mat[i][j] >= -450 and wave_mat[i][j] < -360):
				wave_freq[0] += 1
			elif(wave_mat[i][j] >= -360 and wave_mat[i][j] < -270):
				wave_freq[1] += 1
			elif(wave_mat[i][j] >= -270 and wave_mat[i][j] < -180):
				wave_freq[2] += 1
			elif(wave_mat[i][j] >= -180 and wave_mat[i][j] < -90):
				wave_freq[3] += 1
			elif(wave_mat[i][j] >= -90 and wave_mat[i][j] < 0):
				wave_freq[4] += 1
			elif(wave_mat[i][j] >= 0 and wave_mat[i][j] < 90):
				wave_freq[5] += 1
			elif(wave_mat[i][j] >= 90 and wave_mat[i][j] < 180):
				wave_freq[6] += 1
			elif(wave_mat[i][j] >= 180 and wave_mat[i][j] < 270):
				wave_freq[7] += 1
			elif(wave_mat[i][j] >= 270 and wave_mat[i][j] < 360):
				wave_freq[8] += 1
			elif(wave_mat[i][j] >= 360 and wave_mat[i][j] < 450):
				wave_freq[9] += 1
	print 'wave_freq', wave_freq
	return wave_freq 								# 将频数矩阵返回

# 将执行步骤编写成一个函数，方便其他模块调用
def calcWaveFeature(img_dir, img_path, fea_dir):
	wave_feature = {}
	i = 0											# 用来显示程序的执行进度
	# print len(img_path)
	for path in img_path:							# 遍历每张图片
		split_wave_feature = list()					# 存放分割图片的小波特征
		# print path
		i += 1
		# print i, '/', len(img_path)					# 显示执行进度
		img = cv2.imread(img_dir + path, 0)			# 以灰度形式读取图像原文件
		# img = cv2.imread('./test/train/3/30001.jpg', 0)	# 以灰度形式读取图像源文件
		# getWaveCoe(img)									# 对原图像进行三层小波变换
		split_imgs = preProcess.splitImg(img)
		j = 0
		for split_img in split_imgs:
			j += 1
			print i, '/', len(img_path), '/', j
			split_wave_feature.extend(getWaveCoe(split_img))
		# print len(split_wave_feature)
		# print split_wave_feature
		# 将小波系数频数保存在字典中，设置相应的键值
		wave_feature[path.split('.')[0]] = split_wave_feature	
	# 将字典以.pkl后缀名的形式存储在文件中
	pickle.dump(wave_feature, open(fea_dir + 'wave_feature.pkl', 'wb'))	


if __name__ == '__main__':
	img_path = preProcess.getImgPath()
	pre_path = './test/train/'
	calcWaveFeature(pre_path, img_path, './split_features/')
	


