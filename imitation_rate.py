#imitation rate count
import csv
import os
import sys
f = sys.argv

def imitation_rate(fi,first,second,num):
	with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			a=0
			f = open(fi, 'r')
			for row in csv.reader(csvinput):
				writer.writerow(row+["imitation_rate"])
				break
			prev=0
			for row,line in zip(csv.reader(csvinput),f):
				#print 'd'
				s= open('./chats_process/'+str(first)+'_'+str(second)+'/'+'/imitation_rate_'+str(num)+'.txt','a')
				count = 0
				
				
				if line.find("<Media omitted>")==-1:
					
					p=line.split()
					u = int(p[2])
					if prev==0:
						s.write(str(a) + " " + str(0))
						writer.writerow(row+["0"])
					else:	
						b = float(u)/prev						
						s.write(str(a+1) + " " + str(b))
						writer.writerow(row+[b])
					prev=u
					s.write("\n")
				else:
					s.write(str(a) + " -> " + str(0))
					writer.writerow(row+[0])
					s.write("\n")
					
				
				a = a+1
				s.close()
			f.close()
			
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')

if __name__ == "__main__":
	imitation_rate(str(f[1]),str(f[2]),str(f[3]),0)
