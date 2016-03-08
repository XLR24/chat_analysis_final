import csv
import random
import math
import os
import sys
from sklearn import svm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
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
def convert_float(copy):
	for i in range(len(copy)):
		copy[i] = [float(x) for x in copy[i]]
	return copy
def main():
	path = './chats_process'
	m = "test_negative.csv"
	test_nega = loadCsv(m)
	
	for filename in os.listdir(path):
		filename_copy = filename
		filename=filename.split("_")
		first=filename[0]
		second=filename[1]
		print filename
		
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_1_final.csv','r') as f:
			with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'single_class_ml_1.csv','w') as f1:
				f.next() 
				for line in f:
					line = line.replace('\n', '').replace('\r', '')
					
					line=line+'\n'
					lines=line.split(',')
					a=0
					linet=''
					for word in lines:
						if a==0:
							a=a+1
							continue
						linet=linet+','+word
		
					
					linet = linet[1:]
						
					#print linet
					f1.write(linet)
			f1.close()
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_2_final.csv','r') as f:
			with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'single_class_ml_2.csv','w') as f1:
				f.next() 
				for line in f:
					line = line.replace('\n', '').replace('\r', '')
					line=line+'\n'

					lines=line.split(',')
					a=0
					linet=''
					for word in lines:
						if a==0:
							a=a+1
							continue
						linet=linet+','+word
		
					
					linet = linet[1:]
						
					#print linet
					f1.write(linet)
			f1.close()
		t = path+'/'+filename_copy+'/single_class_ml_1.csv'
		u = path+'/'+filename_copy+'/single_class_ml_2.csv'

		labels = get_labels(test_nega)
		splitRatio = .5
		dataset = loadCsv(t)
		dataset1 = loadCsv(u)
		
		trainingSet, testSet = splitDataset(dataset, splitRatio)
		'''
		#testSet = testSet + test_nega
		trainingSet1, testSet1 = splitDataset(dataset1, splitRatio)
		testSet1 = testSet1 + test_nega
		'''
		trainingSet = convert_float(trainingSet)
		testSet = convert_float(testSet)
		'''
		trainingSet1= convert_float(trainingSet1)
		testSet1 = convert_float(testSet1)	
		
		#print trainingSet
		#print '\n\n\n'
		#print testSet
		#labels_test = get_labels(testSet)

		xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
		# Generate train data
		#X = 0.3 * np.random.randn(100, 2)
		X_train = trainingSet
		#print trainingSet
		# Generate some regular novel observations
		#X = 0.3 * np.random.randn(20, 2)
		X_test = testSet
		#print testSet
		# Generate some abnormal novel observations
		X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
		# fit the model
		clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
		clf.fit(X_train)
		y_pred_train = clf.predict(X_train)
		y_pred_test = clf.predict(X_test)
		y_pred_outliers = clf.predict(X_outliers)
		n_error_train = y_pred_train[y_pred_train == -1].size
		n_error_test = y_pred_test[y_pred_test == -1].size
		n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
		print float(float(len(trainingSet)-n_error_train)/len(trainingSet))*100
		print float(float(len(testSet)-n_error_test)/len(testSet))*100
		# plot the line, the points, and the nearest vectors to the plane
		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)

		plt.title("Novelty Detection")
		plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
		a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
		plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

		b1 = plt.scatter(X_train[0], X_train[1], c='white')
		b2 = plt.scatter(X_test[0], X_test[1], c='green')
		c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
		plt.axis('tight')
		plt.xlim((-5, 5))
		plt.ylim((-5, 5))
		plt.legend([a.collections[0], b1, b2, c],
			   ["learned frontier", "training observations",
			    "new regular observations", "new abnormal observations"],
			   loc="upper left",
			   prop=matplotlib.font_manager.FontProperties(size=11))
		plt.xlabel(
		    "error train: %d/200 ; errors novel regular: %d/40 ; "
		    "errors novel abnormal: %d/40"
		    % (n_error_train, n_error_test, n_error_outliers))
		plt.show()


		'''
		# SVM
		clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
		clf.fit(trainingSet)
		y_pred_train = clf.predict(trainingSet)
		y_pred_test = clf.predict(testSet)
		X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
		n_error_train = y_pred_train[y_pred_train == -1].size
		n_error_test = y_pred_test[y_pred_test == -1].size
		print float(float(len(trainingSet)-n_error_train)/len(trainingSet))*100
		print float(float(len(testSet)-n_error_test)/len(testSet))*100
		'''
		clf = svm.OneClassSVM(probability=True)
		clf.fit(train_set)
		#clf.decision_function(test_set)
		results_SVM = clf.predict(test_set)
		a = clf.predict_proba(test_set)
		acc_svm = getAccuracy(results_SVM,labels_test)
		print "accuracy_svm= " + str(acc_svm)
		'''
main()
