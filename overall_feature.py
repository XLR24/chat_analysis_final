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
				writer.writerow(row+["previous_stopwords","previous_punctuation","previous_frequent"])
				break
			
			
			t = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_stopwords_'+str(num)+'.txt','r')
			u = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_punctuation_'+str(num)+'.txt','r')
			#v = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_acrnonyms_'+str(num)+'.txt','r')
			#w = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_stopwords'+str(num)+'.txt','r')
			x = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'freq_dist_'+str(num)+'.txt','r')
			
			'''
			u = open("./package/previous_punctuation_1.txt",'r')
			v = open("./package/previous_acrnonyms_1.txt",'r')
			w = open("./package/smiley.txt",'r')
			x = open("./package/freq_dist.txt",'r')
			'''

			stopword = defaultdict(int)
			punctuation = defaultdict(int)
			#acronym = defaultdict(int)
			frequent_words =defaultdict(int)
			#smiley = defaultdict(int)
				
			for row in t:
				#print row
				a = row.split()
				stopword[a[0]]=a[1]
	
			for row in u:
				#print row
				a = row.split()
				punctuation[a[0]]=a[1]
			'''
			for row in v:
				#print row
				a = row.split()
				acronym[a[0]]=a[1]
			'''
			for row in x:
				#print row
				a = row.split()
				frequent_words[a[0]]=a[1]
			'''
			for row in w:
				#print row
				a = row.split()
				smiley[a[0]]=a[1]
			'''

			f = open(fi, 'r')
			
			for row,line in zip(csv.reader(csvinput),f):	
				
				score=0
				score1=0
				score2=0
				score3=0
				score4=0
				line = line.split()
				
				for word in line:
					if stopword.has_key(word):
						#print d1[word]
						#print "heo"
						score+=int(stopword[word])
					if punctuation.has_key(word):
						score1+=int(punctuation[word])
					'''
					if acronym.has_key(word):
						score2+=int(acronym[word])
					
					if smiley.has_key(word):
						score3+=int(punctuation[word])
					'''

					if frequent_words.has_key(word):
						score4+=int(frequent_words[word])

				writer.writerow(row+[str(score),str(score1),str(score4)])

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
		score=0
		score1=0
		score2=0
		score3=0
		score4=0
		line = line.split()
		for word in line:
			if stopword.has_key(word):
				#print d1[word]
				#print "heo"
				score+=int(stopword[word])
			if punctutaion.has_key(word):
				score1+=int(punctuation[word])
			if acronym.has_key(word):
				score2+=int(acronym[word])
			if smiley.has_key(word):
				score3+=int(punctuation[word])
			if frequent_words.has_key(word):
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
	
