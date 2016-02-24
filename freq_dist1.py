from collections import defaultdict
import csv
import os
import sys
f = sys.argv
result = defaultdict(int)

def freq_words(fi,first,second,num):
	
	f = open(fi, 'r')

	for line in f:
		line=line.split()
    
		for word in line:
			result[word] +=1			

	s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'freq_dist_'+str(num)+'.txt','wb')

	for key, value in result.items():
		s.write(str(key) + " -> " + str(value))
		s.write("\n")


		
