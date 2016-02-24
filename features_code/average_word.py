#filtering the chat
import csv
import os
import sys

def average_word(fi,first,second,num):
	with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			a=0
			f = open(fi, 'r')
			for row,line in zip(csv.reader(csvinput),f):	
				s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'number_word_'+str(num)+'.txt','a')
				count = 0
				space = 0
				if a==0:
					writer.writerow(row+["average_word"])
				else:
					if line.find("<Media omitted>")==-1:
						for t in line:
							if(t==' '):
								space=space+1
						s.write(str(a) + " -> " + str(space))
						
						writer.writerow(row+[space])
						s.write("\n")
					else:
						s.write(str(a) + " -> " +"0")
						
						writer.writerow(row+["0"])
						s.write("\n")
						
				a = a+1
				

			s.close()
			f.close()
			
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')

f = sys.argv
if __name__ == "__main__":
	average_word(str(f[1]),str(f[2]),str(f[3]),0)
