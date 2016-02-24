#filtering the chat
import csv
import os
import sys
from collections import defaultdict

f = sys.argv


def acro_line(fi,first,second,num):
	with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			
			a=0
			f = open(fi, 'r')
			f1 = open('./package/acro.txt', 'r')
			l=[]
			for line in f1:
				a=line.find(":")
				l.append(line[:a-1])

			a=0
			for row,line in zip(csv.reader(csvinput),f):	
				
				s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'acro_count_'+str(num)+'.txt','a')
				count = 0

				if a==0:
					writer.writerow(row+["acro_line"])
				else:
					if line.find("<Media omitted>")==-1:
						
						line = line.split()
						for word in line:
							for w in l:
								if w.lower() == word.lower():
									count+=1


						s.write(str(a) + " -> " + str(count))
						writer.writerow(row+[count])
						s.write("\n")
						count=0

					else:

						s.write(str(a) + " -> " +"0")	
						writer.writerow(row+["0"])
						s.write("\n")
						
				a = a+1
				

			s.close()
			f.close()
			
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')

if __name__ == "__main__":
	acro_line(str(f[1]),str(f[2]),str(f[3]),0)



