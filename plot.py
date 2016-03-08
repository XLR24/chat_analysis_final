#input - txt file containing list in every line 
#a line graph is to be plotted using each list as a line

import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import math
import os
import sys
x1 = [0,1,2,3,4,5]
LABELS= ['simple_nb','svm','KNN','gausian_nb','bernoulli','random_forest']
plt.title("Accuracy of different algorithm on different feature")
plt.xlabel("Algorithms used")
plt.ylabel("Accuracy")
f = sys.argv
line_label = ['average_word','word_length','uppercase','lowercase','smiley_count','stopword','punct','msg_length','acronym','elongation_vowel','suspension','imita','word_length','media_url','prev_stop','prev_aro','prev_punct','prev_smiley','prev_freq','writitng_speed']
#print f
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
		
	print label
	
	plt.plot(x1,res,marker='o',label =label)
	plt.legend(bbox_to_anchor=(1,1), loc=2, borderaxespad=0.)

plt.xticks(x1, LABELS) 	
plt.show()
