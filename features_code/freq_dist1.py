from collections import defaultdict
import csv
import os
import sys
f = sys.argv
result = defaultdict(int)
def freq_words(fi,first,second,num):
			
	for line in fi:
	    	words = line.split()
		for word in words:
		result[word] +=1			

	s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'freq_dist_1.txt','wb')

	for key, value in result.items():
		s.write(str(key) + " -> " + str(value))
		s.write("\n")


		
