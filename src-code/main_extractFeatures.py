# -*- coding: utf-8 -*-

'''
	Date:15/05/2017
	Author:NewCoderQ
	Function:the main function of extract train image features
'''

import preProcess			# 一些预处理的操作
import extractColorFeature	# 颜色特征
import extractGLCMFeature	# GLCM
import extractWaveFeature	# wave feature

img_pre_path = './test/train/'
save_feature_path = './split_features/'
img_path = preProcess.getImgPath()		# the file path of train images

# 颜色特征
extractColorFeature.getGrayCount(img_pre_path, img_path, save_feature_path)

# GLCM特征
extractGLCMFeature.calcGLCMFeature(img_pre_path, img_path, save_feature_path)

# wave feature
extractWaveFeature.calcWaveFeature(img_pre_path, img_path, save_feature_path)


