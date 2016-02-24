#filtering the chat
import csv
import os
import sys

def ratio(fi,first,second,num):
	with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			f = open(fi, 'r')
			a=0
			
			for row in csv.reader(csvinput):
				writer.writerow(row+["up","low_ratio"])
				break
					
			for row,line in zip(csv.reader(csvinput),f):	
				s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'upp_low_ratio'+str(num)+'.txt','a')

				upp_count = 0
				low_count = 0
				
				
				for t in line:
					if t>='a' and t<='z':
						low_count=low_count+1
					if t>='A' and t<='Z':
						upp_count=upp_count+1
				s.write(str(a) + " -> " + str(low_count)+"/"+str(upp_count))
				writer.writerow(row+[low_count,upp_count])
				s.write("\n")
				a = a+1
			s.close()
			f.close()
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')

f = sys.argv
if __name__ == "__main__":
	ratio(str(f[1]),str(f[2]),str(f[3]),0)
