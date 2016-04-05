# to incorporate the previously used individual features 
from collections import defaultdict
import os
import sys

import csv

def overall_feature(fi,first,second,num):
	
	with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			a=0

			for row in csv.reader(csvinput):
				writer.writerow(row+["previous_stopwords","prev_punct","previous_acronym","freq_word","prev_smiley"])
				break
			
			
			t = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_stopwords_'+str(num)+'.txt','r')
			u = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_punctuation_'+str(num)+'.txt','r')
			v = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_acrnonyms_'+str(num)+'.txt','r')
			w = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_smiley_'+str(num)+'.txt','r')
			x = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'freq_dist_'+str(num)+'.txt','r')
			
		

			stopword = defaultdict(int)
			punctuation = defaultdict(int)
			acronym = defaultdict(int)
			frequent_words =defaultdict(int)
			smiley = defaultdict(int)
				
			for row in t:
				#print row
				a = row.split()
				stopword[a[0]]=a[1]
	
			for row in u:
				#print row
				a = row.split()
				punctuation[a[0]]=a[1]
			
			for row in v:
				#print row
				a = row.split()
				acronym[a[0]]=a[1]
			for row in w:
				#print row
				a = row.split()
				smiley[a[0]]=a[1]	
			for row in x:
				#print row
				a = row.split()
				
				if a[0] in punctuation or a[0] in stopword or a[0] in acronym or a[0] in smiley:
					continue
				else:
					frequent_words[a[0]]=a[1]
			
		
			
			#print frequent_words
			f = open(fi, 'r')
			count=1
			for row,line in zip(csv.reader(csvinput),f):	
				
				score=0
				score1=0
				score2=0
				score3=0
				score4=0
			
				for p in line:
					if punctuation.has_key(p):
						score1+=int(punctuation[p])
						#print str(count) + p + " " + str(score1)
				
				line1=unicode(line,"utf-8")
				line = line.split()
				line1=line1.split()
				count+=1
				for word in line1:
					for letter in word:
							st=letter.encode('utf-8')
							if smiley.has_key(st):
								score4+=int(smiley[st])
								#print str(count) + st + " " + str(score4)
				for word in line:
					
				
					if stopword.has_key(word):
						#print d1[word]
						#print "heo"
						score+=int(stopword[word])
						#print str(count) +" " + str(score)
					
					elif acronym.has_key(word):
						score2+=int(acronym[word])
						#print str(count) +" " + str(score2)
					
					elif frequent_words.has_key(word):
						score3+=int(frequent_words[word])
						#print str(word) +" " + str(score3)
					
					else:
						continue
				
				writer.writerow(row+[str(score),str(score1),str(score2),str(score3),str(score4)])

			t.close()
			u.close()
			x.close()
			f.close()
			
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')




	'''
	s = open(f,'r')

	t = open("./package/previous_stopwords_1.txt",'r')
	u = open("./package/previous_punctuation_1.txt",'r')
	v = open("./package/previous_acrnonyms_1.txt",'r')
	w = open("./package/smiley.txt",'r')
	x = open("./package/freq_dist.txt",'r')
	s_p = open("./package/stopword_score.txt",'a')
	punct = open("./package/punct_score.txt",'a')
	acro = open("./package/acro_score.txt",'a')
	smile = open("./package/smile_score.txt",'a')
	f_w = open("./package/frequent_score.txt",'a')

	stopword = defaultdict(int)
	punctuation = defaultdict(int)
	acronym = defaultdict(int)
	frequent_words =defaultdict(int)
	smiley = defaultdict(int)
	
	for row in t:
		#print row
		a = row.split()
		stopword[a[0]]=a[1]
	
	for row in u:
		#print row
		a = row.split()
		punctuation[a[0]]=a[1]
	
	for row in v:
		#print row
		a = row.split()
		acronym[a[0]]=a[1]
	
	for row in w:
		#print row
		a = row.split()
		frequent_words[a[0]]=a[1]
	
	for row in x:
		#print row
		a = row.split()
		smiley[a[0]]=a[1]

	
	#print d1
	for line in s:
<<<<<<< HEAD
		score=0,score1=0,score2=0,score3=0,score4=0
=======
		score=0
		score1=0
		score2=0
		score3=0
		score4=0
>>>>>>> 74618391f8a0ecf518db4636538a858ca3924f6e
		line = line.split()
		for word in line:
			if stopword.has_key(word):
				#print d1[word]
				#print "heo"
				score+=int(stopword[word])
<<<<<<< HEAD
			if punctutaion.has_key(word)
				score1+=int(punctuation[word])
			if acronym.has_key(word)
				score2+=int(acronym[word])
			if smiley.has_key(word)
				score3+=int(punctuation[word])
			if frequent_words.has_key(word)
=======
			if punctutaion.has_key(word):
				score1+=int(punctuation[word])
			if acronym.has_key(word):
				score2+=int(acronym[word])
			if smiley.has_key(word):
				score3+=int(punctuation[word])
			if frequent_words.has_key(word):
>>>>>>> 74618391f8a0ecf518db4636538a858ca3924f6e
				score4+=int(punctuation[word])
			
		s_p.write(str(score))
		s_p.write("\n")
		punct(str(score1))
		punct.write("\n")
		acro.write(str(score2))
		acro.write("\n")
		f_w.write(str(score3))
		f_w.write("\n")
		smile.write(str(score4))
		smile.write("\n")
	

	'''

f = sys.argv
if __name__ == "__main__":
    overall_feature(str(f[1]),str(f[2]),str(f[3]),0)

	
