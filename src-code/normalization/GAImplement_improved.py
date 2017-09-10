# -*- coding: utf-8 -*-

'''
	Date:22/05/2017
	Author:NewCoderQ
	Function:
		GA:Genetic Algorithm
'''

import random
import numpy as np
import svmTrainGetWeight		# 交叉验证获得每种权重的准确率
import os
import svmDataPreProcess		# 测试数据处理
from svmutil import *			# libsvm tool
import copy						# 完成列表元素的拷贝
import time			


popSize = 50			# 定义种群的数量
ValNum = 3				# 设置变量的个数为3
initpop = list()		# 用来存放基因种群 50 * 3
pop = list()			# 用来存放二进制基因序列，50个元素
pop_Back = list()		# 用来存储父辈的二进制基因序列，50个元素
# 用来存放结果基因的列表元素(实数)
resultList = [0 for j in range(popSize)]	# 定义一个50维的列表
# 备份用来处理不符合范围的变量的数据
resultList_Back = [0 for j in range(popSize)]		# 定义一个50维的列表
MJ2 = 4194304			# 2^22,这个数字用来将十进制的数转换成22位二进制数
M = 26					# 定义每个变量二进制基因序列的长度为22，这样便于后续的交叉与变异
PC = 0.35				# 基因交叉概率
PM = 0.2				# 基因变异概率
fitness = list()		# 用来存放每条基因的适应度
Accuracy = list()		# 用来存放每次加权后的交叉验证的正确率
						# 即个体适应度
# RF = list()				# 个体的相对适应度 列表
	

# initialize the gene generations	50
def initGene(n):
	for i in range(n):
		gene = list()
		for j in range(ValNum):				# 3
			gene.append(random.random() * 10)
		initpop.append(gene)
	
	'''
		将初始化列表赋值给resultList和resutList_Back
		此处要注意二维列表(列表元素为列表的时候)的子列表的赋值方法
	'''
	for m in range(n):			# 50
		# 对列表中的子元素做深拷贝(传值)
		resultList[m] = copy.deepcopy(initpop[m])
		resultList_Back[m] = copy.deepcopy(initpop[m])


# encoding
def encoding():
	for i in range(len(initpop)):				# 50
		binary_str = ""
		for j in range(len(initpop[0])):		# 3
			d1 = initpop[i][j] * (MJ2 - 1)			# 为接下来转换成二进制基因序列做准备
			# 将十进制的数值转换成二进制
			# 二进制转换的结果以字符串的形式返回，替换掉首端的0b
			binary_val = bin(int(d1)).replace('0b', '')
			if(len(binary_val) < M):		# 25
				for k in range(M - len(binary_val)):
					binary_val = '0' + binary_val
			'''
				test code
			
			print initpop[i][j], '-->', int(d1), '-->', binary_val, \
			      '-->', int(binary_val, 2), '-->', (int(binary_val, 2)) / (MJ2 - 1.0)
			# print int(binary_val, 2) / (MJ2 - 1.0)
			'''
			binary_str += binary_val
			# print binary_val
		pop.append(binary_str) # 将binary_str存储在pop列表中
	pop_Back.extend(pop)
	# print pop


# decoding
# 将二进制的基因序列解码转换成十进制
'''
	编码要处理好不符合条件的变量的问题
'''
def decoding():
	fault_index = list()				# 用来存放不符合范围的变量的索引
	index = 0

	for i in range(len(pop)):			# 50
		for j in range(ValNum):			# 3

		# 先将二进制基因序列转换成十进制的基因
			gene_int = int(pop[i][(j * M):((j + 1) * M)], 2) / (MJ2 - 1.0)
			if gene_int < 10 or gene_int == 10:						# 如果基因序列中三个变量均满足0 ~ 10的范围要求
				resultList[i][j] = gene_int							# 就将该基因序列赋值到相应的结果序列中
			else:
				# 判断index列表中是否存在该i值
				if i not in fault_index:					# 如果i不在fault_index中
					fault_index.append(i)					# 就将i添加进fault_index列表中

	'''
		判断Accuracy列表中是否有数据
			如果len(Accuracy) == 0: 说明是第一次调用decoding()函数，则不进行下面的操作
			否则，找出Accuracy列表中最大元素的索引，将其对应的三个变量赋值给交叉变换中不符合要求的基因序列	
	'''

	# print '"""""""""""""""""""""""""""""""""""""""""""""'

	# print 'resultList_Back:'
	# print resultList_Back

	if len(Accuracy) > 0:			# 程序不是第一次进行解码操作
		max_index = Accuracy.index(max(Accuracy))			# 获取最大值的索引
		print 'MAX Accuracy:', max(Accuracy), '%'
		print resultList_Back[max_index]
		for k in range(len(fault_index)):					# 遍历fault_index列表中的每个元素
			# print k
			# 将最大函数值对应的三个变量赋值给交叉变换后不满足条件范围的变量
			resultList[fault_index[k]] = resultList_Back[max_index]		

			pop[k] = pop_Back[max_index]					# 最优二进制基因序列赋值
		
		# 接下来将子代基因序列赋值为父辈基因
		for x in range(popSize):
			resultList_Back[x] = copy.deepcopy(resultList[x])
			pop_Back[x] = pop[x]

	# print 'resultList:'
	# print resultList


# calcuate fitness
def fitness(color_feature, GLCM_feature, wave_feature, color, GLCM, wave, index):	
	log_file = open('Accuracy.log', 'a')
	decoding()							# 解码
	del Accuracy[:]						# 清空列表中的元素，需要进行解码了之后再清空列表
	for i in range(len(resultList)):		# 50
		print '********************', index, '####', (i + 1), '*************************'
		svmTrainGetWeight.generateData_4(color_feature, GLCM_feature, wave_feature, resultList[i])
		# print 'normalize the train_data...'
		# cmd = 'cd 05-17/train_data&svm-scale -l 0 -u 1 -s raw_data svm_data > svm_rawed_data'
		# os.system(cmd)
		# print 'normalization done!'
		print 'data training...'
		svmTrainGetWeight.svmTrain_4()
		'''
			testing...
		'''
		print 'testing...'
		svmDataPreProcess.generateData(color, GLCM, wave, resultList[i])
		# print 'normalize the train_data...'
		# cmd = 'cd 05-17/test_data&svm-scale -l 0 -u 1 -s raw_data test_data > test_rawed_data'
		# os.system(cmd)
		# print 'normalization done!'
		# y, x = svm_read_problem('05-17/test_data/test_rawed_data')
		y, x = svm_read_problem('05-17/test_data/test_data')
		model = svm_load_model('05-17/train_data/model_weight')
		p_label, p_acc, p_val = svm_predict(y, x, model)

		Accuracy.append(p_acc[0])					# 将tuple p_acc的第一个元素(准确率)添加到Accuracy列表中
		log_text = str(resultList[i][0]) + '\t' + str(resultList[i][1]) + '\t' + \
				   str(resultList[i][2]) + '\t' + str(p_acc) + '\n'
		log_file.write(log_text)
		log_file.flush()					# 清理缓存，将缓存区的数据及时写入文件
	log_file.close()


# test code: fitness
def test_fitness():
	decoding()								# 将二进制基因序列解码成十进制
	log_file = open('equation', 'a')		# 用来保存运行日志文件
	del Accuracy[:]							# 清空列表的元素，注意清空列表的位置
	for i in range(len(resultList)):		# 50
		log_text = ''
		equation = 0						# 将结果初始化为0
		for j in range(len(resultList[i])):	# 3
			log_text = log_text + str(resultList[i][j]) + '\t'
			equation += resultList[i][j]
		Accuracy.append(equation)
		log_text = log_text + str(equation) + '\n'
		log_file.write(log_text)
	log_file.close()


# crossover: 单点交叉
# pop: 二进制基因序列列表 50
def crossover():
	for i in range(popSize):				# 50
		cp = random.random()				# 0 - 1.0随机数生成
		if cp < PC:							# 生成的随机数小于交叉概率
			# 生成单点交叉的基因序列
			gene_1 = random.randint(0, popSize - 1)		
			gene_2 = random.randint(0, popSize - 1)
			# 生成单点交叉的交叉索引值
			index_cross = random.randint(0, 3 * M)
			# cross over
			str_11 = ''; str_12 = ''; str_21 = ''; str_22 = ''

			str_11 = pop[gene_1][:index_cross]
			str_12 = pop[gene_1][index_cross:]
			str_21 = pop[gene_2][:index_cross]
			str_22 = pop[gene_2][index_cross:]
			# combine String 
			pop[gene_1] = str_11 + str_22
			pop[gene_2] = str_21 + str_12


# mutation: 基因变异
def mutation():
	for i in range(len(pop)):				# popSize  50
		for j in range(len(pop[i])):		# LENGTH   66
			mp = random.random()			# 生成一个随机数，用来决定是否变异
			if mp < PM:						# 如果生成的随机数小于变异概率
											# 则对该基因位进行变异
				# 对基因位进行变异
				sub_str_1 = pop[i][:j]
				sub_str_2 = pop[i][(j + 1):]
				if pop[i][j] == '0':		# 如果指定的基因位的值为0，则变异为1
					pop[i] = sub_str_1 + '1' + sub_str_2
				else:
					pop[i] = sub_str_1 + '0' + sub_str_2


# roulettewheel: 轮盘赌法进行基因序列的选择
def roulettewheel():
	# 计算所有适应度的和
	sum = 0
	for i in range(popSize):
		sum += Accuracy[i]				


	# 计算相对适应度
	RF = list()								# 存放累计适应度
	for j in range(popSize):
		RF.append(Accuracy[j] / sum)
	# print 'len of RF:', len(RF), RF[49]

	# 计算累计适应度
	c_f = list()							# 用来存放累积适应度
	c_f.append(RF[0])						# 将相对适应度的第一个元素加入到累计适应度中
	for m in range(1, popSize):				# 1 - 50 
		c_f.append(c_f[len(c_f) - 1] + RF[m])		# 计算累积适应度
	# print 'len of c_f:', len(c_f), c_f[49]

	k = 0									# 用来记录选取元素的索引
	tempPop = list()						# 用来临时存放二进制基因序列
	for x in range(popSize):				# 50
		p = random.random()				
		for y in range(popSize):			# 50
			if p < c_f[y]:
				k = y
				break
			else:
				continue
		tempPop.append(pop[k])

	for w in range(popSize):				# 将临时生成的二进制基因序列重新保存在pop列表中
		pop[w] = tempPop[w]

	# print 'len of pop:', len(pop), len(pop[49])


# main function of GA
def GA(n):
	start_time = time.time()				# 开始时间
	color_feature, GLCM_feature, wave_feature = svmTrainGetWeight.loadData()
	color, GLCM, wave = svmDataPreProcess.loadData()
	initGene(popSize)		# 生成初始化的实数基因序列  50
	encoding()
	for i in range(n):				# 设置迭代次数
		each_start_time = time.time()
		if i % 100 == 0:
			print '******************************', (i + 1), '**************************************'
		fitness(color_feature, GLCM_feature, wave_feature, color, GLCM, wave, (i + 1))
		# test_fitness()

		roulettewheel()				# 轮盘赌法选择基因序列
		crossover()					# 二进制基因序列进行单点交叉

		'''
				10101111111
				11111010101

		'''
		mutation()					# 基因变异
		print 'once cost time:', int(time.time() - each_start_time), 's'
	print 'all cost time:', int(time.time() - start_time), 's'


# test code
if __name__ == '__main__':   
	GA(1000)						# 此处将迭代次数设为50