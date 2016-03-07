def main():
	path = './chats_process'
	m = "test_negative.csv"
	test_nega = loadCsv(m)
	for filename in os.listdir(path):
		
		print filename
		t = path+"/"+filename+"/training_set_1_final.csv"
		u = path+"/"+filename+"/training_set_2_final.csv"
		splitRatio = .7
		dataset = loadCsv(t)
		dataset1 = loadCsv(u)
		trainingSet, testSet = splitDataset(dataset, splitRatio)
		testSet = testSet + test_nega
		trainingSet1, testSet1 = splitDataset(dataset1, splitRatio)
		testSet1 = testSet + test_nega

		trainingSet = convert_float(trainingSet)
		testSet = convert_float(testSet)
		
		trainingSet1= convert_float(trainingSet1)
		testSet1 = convert_float(testSe1t)	
		
		
		labels_test = get_labels(test_copy)
		
		# SVM
		clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
		clf.fit(trainingSet)
		y_pred_train = clf.predict(trainingset)
		y_pred_test = clf.predict(testSet)
		
		n_error_train = y_pred_train[y_pred_train == -1].size
		n_error_test = y_pred_test[y_pred_test == -1].size
		print n_error_train
		print n_error_test
"""
		clf = svm.OneClassSVM(probability=True)
		clf.fit(train_set)
		#clf.decision_function(test_set)
		results_SVM = clf.predict(test_set)
		a = clf.predict_proba(test_set)
		acc_svm = getAccuracy(results_SVM,labels_test)
		print "accuracy_svm= " + str(acc_svm)
"""
