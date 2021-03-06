# to get the min,max,average accuracy of all the classifers used till now and write the results in text file namely remove_one3.txt
import csv
import random
import math
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	'''
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	'''
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
 
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	abcd = float(len(numbers)-1)
	if abcd ==0:
		abcd=1
	variance = sum([pow(x-avg,2) for x in numbers])/ abcd
	return math.sqrt(variance)
 
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
 
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries
 
def calculateProbability(x, mean, stdev):
	if stdev==0:
		stdev=0.00000001
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy1(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def convert_float(copy):
	for i in range(len(copy)):
		copy[i] = [float(x) for x in copy[i]]
	return copy


def get_labels(sample_set):

	labels=[]
	labels = [row[-1] for row in sample_set] 	
	for row in sample_set:
		del row[-1]

	'''
	for i in range(len(train_set)):
		train_set[i] = [float(x) for x in train_set[i]]
	'''
	'''for i in range(len(labels)):
		labels[i]=  [float(x) for x in labels[i]]
	'''
	return labels
	#print labels

def seperate1(test_copy):

	labels=[]
	labels = [row[-1] for row in test_copy] 

	for row in test_copy:
		del row[-1]
	'''
	for i in range(len(test_copy)):
		test_copy[i] = [float(x) for x in test_copy[i]]
	'''

	return labels





def main():
	x = [0,1,2,3,4,5]
	LABELS= ['simple_nb','svm','KNN','gausian_nb','bernoulli','random_forest']
	plt.title("Accuracy of different algorithm on different user chat")
	plt.xlabel("Algorithms used")
	plt.ylabel("Accuracy")
	path = './chats_process'
	
	
	#test_negative = convert_float(test_nega)
	#labels_test_negative = get_labels(test_negative)
	
	no_user=0
	
	min1=100
	min2=100
	min3=100
	min4=100
	min5=100
	min6=100
	max1=0
	max2=0
	max3=0
	max4=0
	max5=0
	max6=0
	average = [0,0,0,0,0,0]
	for filename in os.listdir(path):

		no_user+=1
		print filename
		t = path+'/'+filename+'/train.csv'
		splitRatio = .5
		dataset = loadCsv(t)
		trainingSet, testSet = splitDataset(dataset,splitRatio)
		m = 'test_negative.csv'
		test_nega = loadCsv(m)
		testSet = testSet + test_nega
		trainset_copy = trainingSet
		test_copy = testSet

		trainingSet = convert_float(trainingSet)
		testSet = convert_float(testSet)

		#print testSet

		summaries = summarizeByClass(trainingSet)
		predictions = getPredictions(summaries, testSet)
		acc_NB = getAccuracy1(testSet, predictions)
		min1=min(min1,acc_NB)
		max1=max(max1,acc_NB)
		average[0]+=acc_NB

		#print "accuracy_simpleNB= " + str(acc_NB)
		
		train_set = convert_float(trainset_copy)
		labels_train = get_labels(trainset_copy)
		
		test_set = convert_float(test_copy)
		#testSet = testSet + test_negative
		labels_test = get_labels(test_copy)
		#labels_test = labels_test + labels_test_negative
		#print labels_test

		# SVM

		clf = svm.SVC(probability=True)
		clf.fit(train_set, labels_train)
		#clf.decision_function(test_set)
		results_SVM = clf.predict(test_set)
		a = clf.predict_proba(test_set)
		acc_svm = getAccuracy(results_SVM,labels_test)
		#print "accuracy_svm= " + str(acc_svm)
		
		min2=min(min2,acc_svm)
		max2=max(max2,acc_svm)
		average[1]+=acc_svm
		#KNN

		neigh = KNeighborsClassifier(n_neighbors=3)
		neigh.fit(train_set, labels_train)
		results_KNN=neigh.predict(test_set)
		b = neigh.predict_proba(test_set)
		acc_knn = getAccuracy(results_KNN,labels_test)
		#print "accuracy_knn= " + str(acc_knn)
		average[2]+=acc_knn
		min3=min(min3,acc_knn)
		max3=max(max3,acc_knn)
		#gausianNB

		clf = GaussianNB()
		clf.fit(train_set, labels_train)
		results_GausianNB=clf.predict(test_set)
		c = clf.predict_proba(test_set)
		acc_gausNB = getAccuracy(results_GausianNB,labels_test)
		#print "accuracy_gausNB= " + str(acc_gausNB)
		average[3]+=acc_gausNB
		min4=min(min4,acc_gausNB)
		max4=max(max4,acc_gausNB)
		#BernoiliNB

		clf = BernoulliNB()
		clf.fit(train_set, labels_train)
		results_BernoulliNB=clf.predict(test_set)
		d = clf.predict_proba(test_set)
		acc_BernoNB = getAccuracy(results_BernoulliNB,labels_test)
		#print "accuracy_bernoNB= " + str(acc_BernoNB)
		average[4]+=acc_BernoNB
		min5=min(min5,acc_BernoNB)
		max5=max(max5,acc_BernoNB)

		#randomforests

		clf = RandomForestClassifier(n_estimators=10)
		clf.fit(train_set,labels_train)
		results_randomforest=clf.predict(test_set)
		e =  clf.predict_proba(test_set)
		acc_random_F = getAccuracy(results_randomforest,labels_test)
		#print "accuracy_random_forest= " + str(acc_random_F)
		average[5]+=acc_random_F
		min6=min(min6,acc_random_F)
		max6=max(max6,acc_random_F)
		#print "-------------\n"
		#print results_SVM
		#print results_KNN
		#print results_GausianNB
		#print results_BernoulliNB
		#print results_randomforest

		#print "\n"
		#print labels_test
		#print results
	 	#plt.plot(x,results,marker='o')

		'''
		s = open('results.txt','a')

		with open('./chats_process/'+filename+'/'+'ml_training_'+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			for a1,b1,c1,d1,e1,label in zip(a,b,c,d,e,labels_test):
				writer.writerow([a1[1],b1[1],c1[1],d1[1],e1[1],label])
				s.write("%s\n" % a1)
				s.write("%s\n" % b1)
				s.write("%s\n" % c1)
				s.write("%s\n" % d1)
				s.write("%s\n" % e1)
	
				#s.write(b1)
				#s.write(str(c1)) 
				#s.write(d1) 
				#s.write(e1)
				s.write("................\n")

		print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
		# prepare model
		summaries = summarizeByClass(trainingSet)
		# test model
		predictions = getPredictions(summaries, testSet)
		accuracy = getAccuracy(testSet, predictions)
		print('Accuracy: {0}%').format(accuracy) '''
	
	t = open('remove_one4.txt','a')
	t.write(str(min1)+" , " + str(max1)+" , " + str(average[0]/float(no_user))+'\n')
	t.write(str(min2)+" , " + str(max2)+" , " + str(average[1]/float(no_user))+'\n')
	t.write(str(min3)+" , " + str(max3)+" , " + str(average[2]/float(no_user))+'\n')
	t.write(str(min4)+" , " + str(max4)+" , " + str(average[3]/float(no_user))+'\n')
	t.write(str(min5)+" , " + str(max5)+" , " + str(average[4]/float(no_user))+'\n')
	t.write(str(min6)+" , " + str(max6)+" , " + str(average[5]/float(no_user))+'\n')

	#plt.xticks(x, LABELS) 	
	#plt.show()

main()
