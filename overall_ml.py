"""
This file plot the result of combining different ml algorithms for the training and testing set obtained .At first all ml algorithms are trained on a dataset which consiist of positive as well as negative dataset than its probablity score is taken and used for the five different type of estimation or ways of combining.Results are written in combine_ml.txt.
"""
import csv
import random
import math
import os
import sys

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



path = './chats_process'
def main():
	no_user = 0
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
	average = [0,0,0,0,0]
	for filename in os.listdir(path):
		no_user+=1
		print filename
		t = path+"/"+filename+"/train.csv"
		splitRatio = .5
		dataset = loadCsv(t)
		m = 'test_negative.csv'
		test_nega = loadCsv(m)
		trainingSet, testSet = splitDataset(dataset, splitRatio)
		testSet = testSet + test_nega
		trainset_copy = trainingSet
		test_copy = testSet

		trainingSet = convert_float(trainingSet)
		testSet = convert_float(testSet)
		
		
		#print testSet

		summaries = summarizeByClass(trainingSet)
		predictions = getPredictions(summaries, testSet)
		acc_NB = getAccuracy1(testSet, predictions)
		
		print "accuracy_simpleNB= " + str(acc_NB)

		train_set = convert_float(trainset_copy)
		labels_train = get_labels(trainset_copy)

		test_set = convert_float(test_copy)
		
		labels_test = get_labels(test_copy)
		
		# SVM

		clf = svm.SVC(probability=True)
		clf.fit(train_set, labels_train)
		#clf.decision_function(test_set)
		results_SVM = clf.predict(test_set)
		a = clf.predict_proba(test_set)
		acc_svm = getAccuracy(results_SVM,labels_test)
		print "accuracy_svm= " + str(acc_svm)
		#print clf.classes_
		#KNN

		neigh = KNeighborsClassifier(n_neighbors=3)
		neigh.fit(train_set, labels_train)
		results_KNN=neigh.predict(test_set)
		b = neigh.predict_proba(test_set)
		acc_knn = getAccuracy(results_KNN,labels_test)
		print "accuracy_knn= " + str(acc_knn)
		#print neigh.classes_

		#gausianNB

		clf = GaussianNB()
		clf.fit(train_set, labels_train)
		results_GausianNB=clf.predict(test_set)
		c = clf.predict_proba(test_set)
		acc_gausNB = getAccuracy(results_GausianNB,labels_test)
		print "accuracy_gausNB= " + str(acc_gausNB)
		#print clf.classes_
		#BernoiliNB

		clf = BernoulliNB()
		clf.fit(train_set, labels_train)
		results_BernoulliNB=clf.predict(test_set)
		d = clf.predict_proba(test_set)
		acc_BernoNB = getAccuracy(results_BernoulliNB,labels_test)
		print "accuracy_bernoNB= " + str(acc_BernoNB)
		#print clf.classes_
		#randomforests

		clf = RandomForestClassifier(n_estimators=10)
		clf.fit(train_set,labels_train)
		results_randomforest=clf.predict(test_set)
		e =  clf.predict_proba(test_set)
		acc_random_F = getAccuracy(results_randomforest,labels_test)
		print "accuracy_random_forest= " + str(acc_random_F)
		#print clf.classes_
		
		#print results_SVM
		#print results_KNN
		#print results_GausianNB
		#print results_BernoulliNB
		#print results_randomforest

		#print "\n"
		#print labels_test
		s = open('results.txt','a')
		aa=0
		pred_results_average=[]
		pred_results_weight=[]
		pred_results_majority=[]
		pred_results_say=[]

		for a1,b1,c1,d1,e1 in zip(a,b,c,d,e):
			
			say1=max(a1[0],b1[0],c1[0],d1[0],e1[0])
			say2=max(a1[1],b1[1],c1[1],d1[1],e1[1])


			temp1=a1[0]+b1[0]+c1[0]+d1[0]+e1[0]
			temp2=a1[1]+b1[1]+c1[1]+d1[1]+e1[1]
			
			cnt1=0
			cnt2=0

			if(a1[0]>a1[1]):
				cnt1+=1
			else:
				cnt2+=1

			if(b1[0]>b1[1]):
				cnt1+=1
			else:
				cnt2+=1

			if(c1[0]>c1[1]):
				cnt1+=1
			else:
				cnt2+=1

			if(d1[0]>d1[1]):
				cnt1+=1
			else:
				cnt2+=1

			if(e1[0]>e1[1]):
				cnt1+=1
			else:
				cnt2+=1

			temp3=acc_svm*a1[0]+acc_knn*b1[0]+acc_gausNB*c1[0]+acc_BernoNB*d1[0]+acc_random_F*e1[0]
			temp4=acc_svm*a1[1]+acc_knn*b1[1]+acc_gausNB*c1[1]+acc_BernoNB*d1[1]+acc_random_F*e1[1]

			temp1/=5
			temp2/=5

			temp3/=5
			temp4/=5

			if temp1>temp2:
				pred_results_average.append(0)
			else:
				pred_results_average.append(1)	

			if temp3>temp4:
				pred_results_weight.append(0)
			else:
				pred_results_weight.append(1)	

			if cnt1>cnt2:
				pred_results_majority.append(0)
			else:
				pred_results_majority.append(1)

			if say1>say2:
				pred_results_say.append(0)
			else:
				pred_results_say.append(1)

		final_acc=getAccuracy(pred_results_average,labels_test)
		final_acc1=getAccuracy(pred_results_weight,labels_test)
		final_acc2=getAccuracy(pred_results_majority,labels_test)
		final_acc3=getAccuracy(pred_results_say,labels_test)

	
		with open('./chats_process/'+filename+'/'+'ml_training'+'.csv', 'w') as csvoutput:
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
		csvoutput.close()
		m = './chats_process/'+filename+'/'+'ml_training'+'.csv'
		dataset = loadCsv(m)
		training, test = splitDataset(dataset, splitRatio)
		n_set = convert_float(training)
		ls_train = get_labels(training)

		t_set = convert_float(test)
		
		s_test = get_labels(test)
		
		# SVM

		clf = svm.SVC()
		clf.fit(n_set, ls_train)
		#clf.decision_function(test_set)
		results_SVM = clf.predict(t_set)
		
		acc_svm = getAccuracy(results_SVM,s_test)
		min1=min(min1,acc_svm)
		max1=max(max1,acc_svm)
		average[0]+=acc_svm
		print "combining_through_ml= " + str(acc_svm)
		min2=min(min2,final_acc)
		max2=max(max2,final_acc)
		average[1]+=final_acc
		print 'final_Acc_average= '+ str(final_acc)
		min3=min(min3,final_acc1)
		max3=max(max3,final_acc1)
		average[2]+=final_acc1
		print 'final_Acc_weight= '+ str(final_acc1)
		min4=min(min4,final_acc2)
		max4=max(max4,final_acc2)
		average[3]+=final_acc2
		print 'final_Acc_majority= '+ str(final_acc2)
		min5=min(min5,final_acc3)
		max5=max(max5,final_acc3)
		average[4]+=final_acc3
		print 'final_Acc_say= '+ str(final_acc3)
		
		print "-------------\n"

		'''
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
		'''
		'''
		print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
		# prepare model
		summaries = summarizeByClass(trainingSet)
		# test model
		predictions = getPredictions(summaries, testSet)
		accuracy = getAccuracy(testSet, predictions)
		print('Accuracy: {0}%').format(accuracy)
		 	'''
	t = open('combine_ml.txt','a')
	t.write(str(min1)+" , " + str(max1)+" , " + str(average[0]/float(no_user))+'\n')
	t.write(str(min2)+" , " + str(max2)+" , " + str(average[1]/float(no_user))+'\n')
	t.write(str(min3)+" , " + str(max3)+" , " + str(average[2]/float(no_user))+'\n')
	t.write(str(min4)+" , " + str(max4)+" , " + str(average[3]/float(no_user))+'\n')
	t.write(str(min5)+" , " + str(max5)+" , " + str(average[4]/float(no_user))+'\n')



main()

   
