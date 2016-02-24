import os
from collections import defaultdict

f = open('acro.txt', 'r')
f1 = open('1_chat.txt')
f2 = open('2_chat.txt')

a=0
s1 = open('acro_count_1.txt','wb')
s2 = open('acro_count_2.txt','wb')

l=[]
 
d1 = defaultdict(int)
d2 = defaultdict(int)

for line in f:
	a=line.find(":")
	l.append(line[:a-1])

for line in f1:
	line = line.split()
	for word in line:
		for w in l:
			if w.lower() == word.lower():
				d1[w]+=1

for line in f2:
	line = line.split()
	for word in line:
		for w in l:
			if w.lower() == word.lower():
				d2[w]+=1

for key, value in d1.items():
	s1.write(str(key) + " -> " + str(value))
	s1.write("\n")

for key, value in d2.items():
	s2.write(str(key) + " -> " + str(value))
	s2.write("\n")





