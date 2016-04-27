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
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

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


def TruePositive(predictions, testSet):
	correct = 0
	
	for i in range(len(testSet)):
		if testSet[i]==1:
			if testSet[i] == 1 and predictions[i]== 1:
				correct += 1
		else:
			if testSet[i] == 0 and predictions[i]== 0:
				correct += 1
	return correct

def TrueNegative(predictions, testSet):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i] == 1 and predictions[i]== 0:
			correct += 1
	return correct

tpr =[]
fpr =[]

def ROC(predictions, testSet):
	tp = 0
	fp=0
	p = sum(testSet)
	q = len(testSet)-p
	for i in range(len(testSet)):
					
		if testSet[i] == 0 and predictions[i]== 1:
			fp += 1
		if testSet[i] == 1 and predictions[i]== 1:
			tp += 1
		
	tpr.append(tp/float(p))
	fpr.append(fp/float(q))
	

def FalseNegative(predictions, testSet):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i]==1:
			if testSet[i] == 1 and predictions[i]== 1:
				correct += 1
		else:
			if testSet[i] == 0 and predictions[i]== 0:
				correct += 1
	return correct

lab_test1=[]
pred_test1=[]

lab_test2=[]
pred_test2=[]

lab_test3=[]
pred_test3=[]

lab_test4=[]
pred_test4=[]

lab_test5=[]
pred_test5=[]

def main():
	
	path = './chats_process'
	
	overall_prediction = []
	overall_testSet = []
	#test_negative = convert_float(test_nega)
	#labels_test_negative = get_labels(test_negative)
	count=0
	results = [0,0,0,0,0,0]
	for filename in os.listdir(path):
			
		count+=1
	
		#print filename
		t = path+'/'+filename+'/train.csv'
		splitRatio = .5
		dataset = loadCsv(t)
		trainingSet, testSet = splitDataset(dataset, splitRatio)
	
		#testSet = testSet + test_nega
		trainset_copy = trainingSet
		test_copy = testSet
	
		trainingSet = convert_float(trainingSet)
		testSet = convert_float(testSet)

		#print testSet

		summaries = summarizeByClass(trainingSet)
		predictions = getPredictions(summaries, testSet)
		acc_NB = getAccuracy1(testSet, predictions)

		#print "accuracy_simpleNB= " + str(acc_NB)
		results[0]+=acc_NB
		train_set = convert_float(trainset_copy)
		labels_train = get_labels(trainset_copy)

		test_set = convert_float(test_copy)
		#testSet = testSet + test_negative
		labels_test = get_labels(test_copy)
		#labels_test = labels_test + labels_test_negative
		#print labels_test
		'''
		tp_NB=TruePositive(predictions,labels_test)
		tn_NB=TrueNegative(predictions,labels_test)
		fp_NB=FalsePositive(predictions,labels_test)
		fn_NB=FalseNegative(predictions,labels_test)
		

		prec_NB= tp_NB/(tp_NB+fp_NB)
		rec_NB=tp_NB/(tp_NB+fn_NB)
		'''
		# SVM
		clf = svm.SVC(probability=True)
		clf.fit(train_set, labels_train)
		#clf.decision_function(test_set)
		results_SVM = clf.predict(test_set)
		a = clf.predict_proba(test_set)[:,1]
		acc_svm = getAccuracy(results_SVM,labels_test)
		#print "accuracy_svm= " + str(acc_svm)
		results[1]+=acc_svm
		
		lab_test2.extend(labels_test)
		pred_test2.extend(a)	

		'''
		tp_SVM=TruePositive(results_SVM,labels_test)
		tn_SVM=TrueNegative(results_SVM,labels_test)
		fp_SVM=FalsePositive(results_SVM,labels_test)
		fn_SVM=FalseNegative(results_SVM,labels_test)
		
		prec_SVM= tp_SVM/(tp_SVM+fp_SVM)
		rec_SVM=tp_SVM/(tp_SVM+fn_SVM)
		'''
		#KNN
		neigh = KNeighborsClassifier(n_neighbors=3)
		neigh.fit(train_set, labels_train)
		results_KNN=neigh.predict(test_set)
		b = neigh.predict_proba(test_set)[:,1]
		acc_knn = getAccuracy(results_KNN,labels_test)
		#print "accuracy_knn= " + str(acc_knn)
		results[2]+=acc_knn
		
		lab_test3.extend(labels_test)
		pred_test3.extend(b)	

		'''
		tp_knn=TruePositive(results_KNN,labels_test)
		tn_knn=TrueNegative(results_KNN,labels_test)
		fp_knn=FalsePositive(results_KNN,labels_test)
		fn_knn=FalseNegative(results_KNN,labels_test)
		
		prec_knn= tp_knn/(tp_knn+fp_knn)
		rec_knn=tp_knn/(tp_knn+fn_knn)
'''
		#gausianNB
		clf = GaussianNB()
		clf.fit(train_set, labels_train)
		results_GausianNB=clf.predict(test_set)
		c = clf.predict_proba(test_set)[:,1]
		acc_gausNB = getAccuracy(results_GausianNB,labels_test)
		#print "accuracy_gausNB= " + str(acc_gausNB)
		results[3]+=acc_gausNB

		lab_test4.extend(labels_test)
		pred_test4.extend(c)	
		'''
		tp_gnb=TruePositive(results_GausianNB,labels_test)
		tn_gnb=TrueNegative(results_GausianNB,labels_test)
		fp_gnb=FalsePositive(results_GausianNB,labels_test)
		fn_gnb=FalseNegative(results_GausianNB,labels_test)
		
		prec_gnb= tp_gnb/(tp_gnb+fp_gnb)
		rec_gnb=tp_gnb/(tp_gnb+fn_gnb)
'''
		#BernoiliNB
		clf = BernoulliNB()
		clf.fit(train_set, labels_train)
		results_BernoulliNB=clf.predict(test_set)
		d = clf.predict_proba(test_set)[:,1]
		acc_BernoNB = getAccuracy(results_BernoulliNB,labels_test)
		#print "accuracy_bernoNB= " + str(acc_BernoNB)
		results[4]+=acc_BernoNB

		lab_test5.extend(labels_test)
		pred_test5.extend(d)	

		'''
		tp_gnb=TruePositive(results_BernoulliNB,labels_test)
		tn_gnb=TrueNegative(results_BernoulliNB,labels_test)
		fp_gnb=FalsePositive(results_BernoulliNB,labels_test)
		fn_gnb=FalseNegative(results_BernoulliNB,labels_test)
		
		prec_bnb= tp_gnb/(tp_gnb+fp_gnb)
		rec_bnb=tp_gnb/(tp_gnb+fn_gnb)
		'''

		#randomforests

		clf = RandomForestClassifier(n_estimators=10)
		clf.fit(train_set,labels_train)
		results_randomforest=clf.predict(test_set)
		preds =  clf.predict_proba(test_set)[:,1]
		acc_random_F = getAccuracy(results_randomforest,labels_test)
		#print "accuracy_random_forest= " + str(acc_random_F)
		results[5]+=acc_random_F
		
		#ROC(results_randomforest,labels_test)
		lab_test1.extend(labels_test)
		pred_test1.extend(preds)		

		

		'''
		tp_gnb=TruePositive(results_randomforest,labels_test)
		tn_gnb=TrueNegative(results_randomforest,labels_test)
		fp_gnb=FalsePositive(results_randomforest,labels_test)
		fn_gnb=FalseNegative(results_randomforest,labels_test)
		
		prec_rf= tp_gnb/(tp_gnb+fp_gnb)
		rec_rf=tp_gnb/(tp_gnb+fn_gnb)
		'''
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
		print('Accuracy: {0}%').format(accuracy)
	overall_prediction = overall_prediction[1:100]
	overall_testSet = overall_testSet[1:100] '''
	fpr1, tpr1, _ = metrics.roc_curve(lab_test1, pred_test1)	
	fpr2, tpr2, _ = metrics.roc_curve(lab_test2, pred_test2)	
	fpr3, tpr3, _ = metrics.roc_curve(lab_test3, pred_test3)	
	fpr4, tpr4, _ = metrics.roc_curve(lab_test4, pred_test4)	
	fpr5, tpr5, _ = metrics.roc_curve(lab_test5, pred_test5)		

	#FalsePositive(overall_prediction, overall_testSet)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	
	#fpr, tpr, _ = roc_curve( overall_testSet,overall_prediction)
	
	plt.plot(fpr1, tpr1, color='red', linewidth=2, label='random_forest')
	plt.plot(fpr2, tpr2, color='green', linewidth=2, label='svm')
	plt.plot(fpr3, tpr3, color='blue', linewidth=2, label='knn')
	plt.plot(fpr4, tpr4, color='yellow', linewidth=2, label='Gausian_NB')
	plt.plot(fpr5, tpr5, color='brown', linewidth=2, label='Bernouli_NB')
	plt.legend(loc="lower right")
	plt.show()
	'''
	t = open('remove_one5.txt','a')
	t.write(str(prec_NB)+" , " + str(rec_NB)+'\n')
	t.write(str(prec_SVM)+" , " + str(rec_SVM)+'\n')
	t.write(str(prec_gnb)+" , " + str(rec_gnb)+'\n')
	t.write(str(prec_bnb)+" , " + str(rec_bnb)+'\n')
	t.write(str(prec_rf)+" , " + str(rec_rf)+'\n')
	t.write(str(prec_knn)+" , " + str(rec_knn)+'\n')
		'''
	#plt.xticks(x, LABELS) 	
	#plt.show()

main()
