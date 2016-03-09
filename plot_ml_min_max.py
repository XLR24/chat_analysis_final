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
LABELS= ['ml_algo','average','weighted_avg','majority_voting','max_min_algo']
plt.title("Diffrent ways of combining ml algos result")
plt.xlabel("Different ways")
plt.ylabel("Accuracy")
f = sys.argv
line_label = ['min','max']#print f
result = open(f[1],'r')
#plt.plot(x,[1,2,3,4,5,6],marker='o')
minn= []
maxx=[]
for line1 in zip(result):
	
	
	line1 =  line1[0]
	line1 = line1.split(",")
	minn.append(float(line1[0]))
	maxx.append(float(line1[1]))
		
	
print minn
print maxx	
plt.plot(x1,minn,marker='o',label ='min')
plt.plot(x1,maxx,marker='o',label ='max')
plt.fill_between(x1, minn, maxx, color='orange', alpha='0.5')
plt.legend(bbox_to_anchor=(1,1), loc=2, borderaxespad=0.)

plt.xticks(x1, LABELS) 	
plt.show()
