# -*- coding: utf-8 -*-

'''
	Date:17/05/2017
	Author:NewCoderQ
	Function:calculate the accuracy of each category
'''

import pickle

# calculate the accuracy of each category
'''
	parameter:
		p_label:predict label list (float)
'''
def calcAcc(p_label):
	name_label = pickle.load(open('img_name_label.pkl', 'rb'))
	acc_normal = 0.0; acc_slight = 0.0
	acc_moderate = 0.0; acc_severe = 0.0
	current_keys = pickle.load(open('./predict_features/color_feature.pkl', 'rb')).keys()
	sum_predict = 0
	sum_normal = 0; sum_cur_normal = 0
	sum_slight = 0; sum_cur_slight = 0
	sum_moderate = 0; sum_cur_moderate = 0
	sum_severe = 0; sum_cur_severe = 0
	for key in current_keys:
		current_label = name_label[key]
		if current_label == 0:
			sum_normal += 1
			if 0 == int(p_label[sum_predict]):
				sum_cur_normal += 1
		elif current_label == 1:
			sum_slight += 1
			if 1 == int(p_label[sum_predict]):
				sum_cur_slight += 1
		elif current_label == 2:
			sum_moderate += 1
			if 2 == int(p_label[sum_predict]):
				sum_cur_moderate += 1
		elif current_label == 3:
			sum_severe += 1
			if 3 == int(p_label[sum_predict]):
				sum_cur_severe += 1
		sum_predict += 1
		sum_cur = sum_cur_normal + sum_cur_slight + sum_cur_moderate + sum_cur_severe
	print 'the accuracy is: ', sum_cur_normal / float(sum_normal), \
							   sum_cur_slight / float(sum_slight),\
							   sum_cur_moderate / float(sum_moderate),\
							   sum_cur_severe / float(sum_severe)
	print sum_cur / float(sum_predict)
	print 'normal:\tsum:%d, \tcurrect:%d' % (sum_normal, sum_cur_normal)
	print 'slight:\tsum:%d, \tcurrect:%d' % (sum_slight, sum_cur_slight)
	print 'moderate:\tsum:%d, \tcurrect:%d' % (sum_moderate, sum_cur_moderate)
	print 'severe:\tsum:%d, \tcurrect:%d' % (sum_severe, sum_cur_severe)

if __name__ == '__main__':
	p = list()
	calcAcc(p)

