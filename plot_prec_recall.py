#input - txt file containing list in every line 
#a line graph is to be plotted using each list as a line

import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import math
import os
import sys
x1 = [1,2,3,4,5]
LABELS= ['svm','KNN','gausian_nb','bernoulli','random_forest']
plt.title("Precision of different algos")
plt.xlabel("Algorithms used")
plt.ylabel("Precision")
f = sys.argv
line_label = ['min','average','max']#print f
result = open(f[1],'r')
#plt.plot(x,[1,2,3,4,5,6],marker='o')

for line,label  in zip(result,line_label):
	res = [] 
	line1 = line[1:]
	line1 = line1[:len(line1)-2]
	line1 = line1.split(",")
	
	for x in line1:
		#print x
		
		
		res.append(float(x))
		
	
	
	plt.plot(x1,res,marker='o',label =label)
	plt.legend(bbox_to_anchor=(1,1), loc=2, borderaxespad=0.)

plt.xticks(x1, LABELS) 	
plt.show()
