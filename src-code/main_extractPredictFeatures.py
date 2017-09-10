# -*- coding: utf-8 -*-

'''
	Date:12/05/2017
	Author:NewCoderQ
	Function:extract the features of images which will be predicted
			 deal with the relationship between image-name and label
'''

import os					# directory operation
import pickle				# file operation
import extractColorFeature	# color
import extractGLCMFeature	# GLCM
import extractWaveFeature	# wave


# get image filepath
# calculate features
def calcFeatures():
	# get images filepath
	imgPaths = os.listdir('./test/val')
	
	# extract color feature
	extractColorFeature.getGrayCount('./test/val/', imgPaths[:-1], './split_predict_features/')

	# extract GLCM feature
	extractGLCMFeature.calcGLCMFeature('./test/val/', imgPaths[:-1], './split_predict_features/')

	# extract wave feature
	extractWaveFeature.calcWaveFeature('./test/val/', imgPaths[:-1], './split_predict_features/')


# match the name of images with the label
def matchNameLabel():
	img_label = {}
	f = open('./test/val.txt', 'r')
	lines = f.readlines()
	for line in lines:
		line = line.strip()
		img_label[line.split(' ')[0].split('.')[0]] = int(line.split(' ')[1])
	pickle.dump(img_label, open('./split_img_name_label.pkl', 'wb'))


if __name__ == '__main__':
	calcFeatures()
	matchNameLabel()
