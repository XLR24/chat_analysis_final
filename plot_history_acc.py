#input - txt file containing list in every line 
#a line graph is to be plotted using each list as a line

import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import math
import os
import sys
x1 = []

for i in range(7,147,7):
	x1.append(i)

#LABELS= ['simple_nb','svm','KNN','gausian_nb','bernoulli','random_forest']
plt.title("Accuracy of different algos with respect to no of line")
plt.xlabel("No of Lines")
plt.ylabel("Accuracy")
f = sys.argv
line_label = ['SimpleNB','SVM','KNN','GausianNB','BernouliNB','RandomForest']
#print f

#plt.plot(x,[1,2,3,4,5,6],marker='o')
algos=[]
for i in range(0,6):
	result = open(f[1],'r')
	
	res = [] 
	for line in result:
		
		line1 = line[1:]
		line1 = line1[:len(line1)-2]
		line1 = line1.split(",")
		
		res.append(float(line1[i]))
	
	#print res	
	algos.append(res)
	result.close()
#print algos

for algo,label in zip(algos,line_label):
	
	#print algo,x1
	plt.plot(x1,algo,marker='o',label =label)
	plt.legend(bbox_to_anchor=(1,1), loc=2, borderaxespad=0.)
	
plt.show()
