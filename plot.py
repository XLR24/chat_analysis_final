#input - txt file containing list in every line 
#a line graph is to be plotted using each list as a line

import matplotlib.pyplot as plt

import numpy as np
import csv
import random
import math
import os
import sys
x1 = np.arange(6)
LABELS= ['simple_nb','svm','KNN','gausian_nb','bernoulli','random_forest']
colors=['grey','red','maroon','chocolate','olive','yellow','crimson','lime','green','teal','aqua','lightblue','deepskyblue','blue','slateblue','purple','magenta','lightpink','lightgreen','black',]
plt.title("Accuracy of different algorithm on different feature")
plt.xlabel("Algorithms used")
plt.ylabel("Accuracy")
f = sys.argv
line_label = [ 'average_word','word_length','uppercase','lowercase','smiley_count','stopword','punct','msg_length','acronym','elongation_vowel','suspension','imitation_rate','media_url','prev_stop','prev_aro','prev_punct','prev_smiley','prev_freq','response_time','overall',]
#print f
result = open(f[1],'r')
#plt.plot(x,[1,2,3,4,5,6],marker='o')

for line,label,colors  in zip(result,line_label,colors):
	res = [] 
	line1 = line[1:]
	line1 = line1[:len(line1)-2]
	line1 = line1.split("," )
	
	for x in line1:
		#print x
		
		
		res.append(float(x))
		
	print label
	

	plt.plot(x1+0.5,res,marker='o',label =label,color=colors)
	plt.legend(bbox_to_anchor=(0.95,0.8), loc=2, borderaxespad=0.2)

plt.xticks(x1+0.5, LABELS) 	
plt.show()
