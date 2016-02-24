import csv
import os
import sys

from collections import defaultdict


def stopword_usage(fi,first,second,num):
    with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			f = open(fi, 'r')
	   	    	a=0
			count=0
			for row in csv.reader(csvinput):
				writer.writerow(row+["stopword_usage"])
				break
			d1 = defaultdict(int)
			p = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_stopwords_'+str(num)+'.txt','a')

		    	stopword = ['a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but','by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']

		    	s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'stopword_usage_'+str(num)+'.txt','a')
		        
		    	for row,line in zip(csv.reader(csvinput),f):       
		        	for t in line.split():
					for be in stopword:
						if t==be:
						#print t,be
							count=count+1
							d1[t]+=1

						    
			        s.write(str(a) + " " + str(count))
				writer.writerow(row+[count])
			        s.write("\n")
			        count=0
				a+=1


			for key, value in d1.items():
				p.write(str(key) + " " + str(value))
				p.write("\n")


			s.close()

			f.close()
				
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')

f = sys.argv
if __name__ == "__main__":
	stopword_usage(str(f[1]),str(f[2]),str(f[3]),0)








