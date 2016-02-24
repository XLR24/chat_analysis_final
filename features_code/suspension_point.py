#COUNTING SUSPENSION POINT PER LINE
import csv
import os
import sys
f = sys.argv

def suspension_point(fi,first,second,num):
	with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			a=0
			f = open(fi, 'r')
			for row,line in zip(csv.reader(csvinput),f):
				s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'suspension_point_count_'+str(num)+'.txt','a')
				count = 0
				if a==0:
					writer.writerow(row+["suspension_point"])
				else:
					if line.find("<Media omitted>")==-1:
						while line.find("...")!=-1:
							p = line.index("...")
							count+=1
							line = line[:p] + line[p+3:] 
						s.write(str(a) + " -> " + str(count))
						writer.writerow(row+[count])
						s.write("\n")
					else:
						s.write(str(a) + " -> " + "0")
						writer.writerow(row+["0"])
						s.write("\n")
				
				a = a+1
				s.close()
			f.close()
			
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')


if __name__ == "__main__":
	suspension_point(str(f[1]),str(f[2]),str(f[3]),0)
