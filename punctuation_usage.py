#filtering the chat
import csv
import os
import sys


from collections import defaultdict
import codecs

def punctuation(fi,first,second,num):
	with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			a=0

			for row in csv.reader(csvinput):
				writer.writerow(row+["punctuation"])
				break
			d1 = defaultdict(int)
			f = open(fi, 'r')
			for row,line in zip(csv.reader(csvinput),f):	
				s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'punctuation_'+str(num)+'.txt','a')
				p = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'previous_punctuation_'+str(num)+'.txt','a')


				count = 0
				space = 0
				
				
				if line.find("<Media omitted>")==-1:


					a=0
					for t in line:
						if a==0:
							a=1
							continue
						if t>='!' and t<='@':
							count=count+1
							d1[t]+=1
					s.write(str(a) + " " + str(count-1))
					s.write("\n")
					writer.writerow(row+[count-1])
				else:
					s.write(str(a) + " " +"0")
					s.write("\n")
					writer.writerow(row+["0"])		
				a = a+1
			for key, value in d1.items():
				p.write(str(key) + " " + str(value))
				p.write("\n")


			s.close()
			f.close()
			
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')



f = sys.argv
if __name__ == "__main__":
	punctuation(str(f[1]),str(f[2]),str(f[3]),0)



