# -*- coding: utf-8 -*-

'''
	提取图像的颜色特征：
		灰度直方图频数和频率
		将特征保存在features下面的.pkl文件中
	python 绘制颜色直方图
'''

import cv2
import os									# 用来批处理文件夹中的文件
import numpy as np
# np.set_printoptions(threshold='nan')
# from matplotlib import pyplot as plt
import pickle								# 数据格式的存储
import preProcess

# get max value and coordinate
def getMax(mat):
	pos_list = list()						# 定义一个图像坐标的列表
	maxValue = 0
	for i in range(mat.shape[0]):			# 行数
		for j in range(mat.shape[1]):		# 列数
			if maxValue < mat[i][j]:
				maxValue = mat[i][j]
				if mat[i][j] == 233:
					pos_list.append([i, j])
	return maxValue, pos_list

# BGR --> Gray
'''
	parameter:
		img_dir:图片文件的上级目录
		img_path:图片文件的具体名称
		fea_dir:图片特征的保存路径的上级目录
'''
def getGrayCount(img_dir, img_path, fea_dir):
	# img = cv2.imread('./test/test-image/1.jpg') 
	histograms = {}
	i = 0
	for path in img_path:
		print(path)
		i = i + 1
		# print path
		split_hist_list = list()		# 用来存放每个小图片的灰度信息
		img = cv2.imread(img_dir + path, 0)		
		# process each splited image
		split_img_list = preProcess.splitImg(img)
		for split_img in split_img_list:
		# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# print img1
			histr = cv2.calcHist([split_img], [0], None, [32], [0, 256])	# 将整个灰度值均分成32等分，减少计算量，节省时间
			histr = histr.flatten()									# 将矩阵转换为一维矩阵
			hist = histr / sum(histr)								# 计算每个灰度值出现的频率
			split_hist_list.extend(hist)
		# 将灰度频率保存在一个字典中，键为图片的名称，值为图片的灰度频率列表
		histograms[path.split('.')[0]] = np.array(split_hist_list) 		
		print(histograms)			

		if i % 50 == 0:
			print "%d / %d" % (i, len(img_path))
			print "the lenth of hist is %d." % len(split_hist_list)
		# print histograms
	# 使用pickle将python数据保存到文件
	# 将特征字典以.pkl后缀名的形式存储在文件中
	pickle.dump(histograms, open(fea_dir +'color_feature.pkl', 'wb'))
	print('Done!')
	
	# 将.pkl文件中的数据重构成python数据
	# color_feature = pickle.load(open(fea_dir + 'color_feature.pkl', 'rb'))
	# print "length of .pkl is", len(color_feature)
	# print color_feature

if __name__ == '__main__':
	# getGrayCount()
	img_path = preProcess.getImgPath()
	getGrayCount('./test/train/', img_path, './split_features/')
