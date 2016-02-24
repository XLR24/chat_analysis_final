#elongation_of_vowel
import csv
import os
import sys

import re

def elongation_vowel(fi,first,second,num):
	with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			a=0
			f = open(fi, 'r')	
			for row in csv.reader(csvinput):
				writer.writerow(row+["elongation_vowel"])
				break
			for row,line in zip(csv.reader(csvinput),f):
				s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'elongation_vowel_count'+str(num)+'.txt','a')
				#print line
				result=0
				
				if line.find("<Media omitted>")==-1:
					cnt=0						
					#print "dhcj"
					regexp = re.compile(r"(.)\1\1+")
					line=line.split()
					for strr in line:
						match = re.search(regexp, strr)
					#result = re.match(r"(.)\1\1+", line)
					#print "hello"
					#print result
						if match:	
							s.write(str(a) + " -> " + match.group(0))
							cnt+=1
							s.write("\n")
						else:
							s.write(str(a) + " -> " + "0")
							s.write("\n")
					writer.writerow(row+[cnt])
				else:
					s.write(str(a) + " -> " + "0")
					writer.writerow(row+["0"])
					s.write("\n")
				
				a = a+1
				s.close()
			f.close()
			
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')

f = sys.argv
if __name__ == "__main__":
	elongation_vowel(str(f[1]),str(f[2]),str(f[3]),0)
