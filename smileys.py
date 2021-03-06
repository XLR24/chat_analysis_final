#to extract the number of smileys in a line
import csv
import os
import sys
from collections import defaultdict
import codecs




def smiley_count(fi,first,second,num):
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
				with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
						writer = csv.writer(csvoutput)
						d1 = defaultdict(int)
						v = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_smiley_'+str(num)+'.txt','a')
						f = open(fi, 'r')
						cnt = 0
						a=0

						s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'smiley_'+str(num)+'.txt','a')
						for row in csv.reader(csvinput):
							writer.writerow(row+["smiley_count"])
							break
						for row,line in zip(csv.reader(csvinput),f):    
						    
								line=unicode(line,"utf-8")

								words = line.split()
								for word in words: 
										for letter in word:
												st=letter.encode('utf-8')
												if st >= '\xF0\x9F\x98\x81' and st <='\xF0\x9F\x99\x8F':
														cnt+=1
														d1[st]+=1
						#if (a%10) == 0: 
							#print str(a)
								s.write(str(a) + " -> " + str(cnt))
								writer.writerow(row+[cnt])
								s.write("\n")
								cnt=0
								a+=1       
						s.close()
						f.close()
						for key, value in d1.items():
							v.write(str(key) + " " + str(value))
							v.write("\n")
						os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')

f = sys.argv
if __name__ == "__main__":
    smiley_count(str(f[1]),str(f[2]),str(f[3]),0)

