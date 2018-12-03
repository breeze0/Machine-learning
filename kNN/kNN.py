# -*- coding: UTF-8 -*-
from numpy import *
import operator

def createDataSet():
	group = array([[1.1,1.0],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels
#分类器
def classify0(inx, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inx, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis =1)#按行相加
	distances = sqDistances**0.5
	sortedDistanceIndex = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistanceIndex[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse = True)#排序数量而不是lable
	return sortedClassCount[0][0]
#处理文件成可操作数据
def file2matrix(filename):
	fr = open(filename)
	arrayLines = fr.readlines()
	numberOfLines = len(arrayLines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector
#归一化数据
def autoNorm(dataSet):
	minVals = dataSet.min(0)#列的最小值，返回数组
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]#数据集的行数
	normDataSet = dataSet - tile(minVals, (m,1))#minVals重复m行,dataSet相减
	normDataSet = normDataSet/tile(ranges, (m,1))#每个特征值除ranges
	return normDataSet, ranges, minVals
#测试算法
def datingClassTest():
	hoRadio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRadio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
		print "the classifer came back with: %d, the real answer is: %d" % (classifierResult,datingLabels[i])
		if (classifierResult != datingLabels[i]):errorCount += 1.0
	print "the total error rate is: %f" % (errorCount/float(numTestVecs))
#实际预测
def classifyPerson():
	resultList = ['not at all', 'in small does', 'in large does']
	percentTats = float(raw_input("percent of time spent playing video games?"))
	ffMiles = float(raw_input("frequent flier miles earned per year?"))
	iceCream = float(raw_input("liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
	print "You will probably like this person: ", resultList[classifierResult -1]