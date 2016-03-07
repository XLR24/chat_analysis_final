#answer_time
import os
import sys
from datetime import datetime
from datetime import timedelta
import time
import csv

def writing_speed_android(fi,first,second,num):
	with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'_final'+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'_final'+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			a=0

			for row in csv.reader(csvinput):
				writer.writerow(row+["writing_speed"])
				break
			f = open(fi, 'r')
			time_txt = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'time.txt', 'r')
			
			for time in time_txt:			
				a = time.index(":::")
				p=time[:a]
				time = time[a:]
				correctDate=False
				try:
					date1 = datetime.strptime(time,"::: %d/%m/%Y, %H:%M \n")
					correctDate=True
				except ValueError:
					correctDate=False
				if correctDate==True:
					date1.strftime("::: %m/%d/%Y, %I:%M %p\n")
				else:
					date1=datetime.strptime(time,"::: %m/%d/%Y, %I:%M %p\n")
				prev = date1
				break
			
			time_txt = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'time.txt', 'r')
			
			for row,line,time in zip(csv.reader(csvinput),f,time_txt):
				
				a = time.index(":::")
				p=time[:a]
				time = time[a:]
				answer_time = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'writing_speed'+str(num)+'.txt','a')
				correctDate=False
				try:
					date1 = datetime.strptime(time,"::: %d/%m/%Y, %H:%M \n")
					correctDate=True
				except ValueError:
					correctDate=False
				if correctDate==True:
					date1.strftime("::: %m/%d/%Y, %I:%M %p\n")
				else:
					date1=datetime.strptime(time,"::: %m/%d/%Y, %I:%M %p\n")
				
				if(date1-prev > timedelta(hours=2)):
					answer_time.write(str(a) + " " + "-1")	
					writer.writerow(row+["-1"])
				else:
					answer_time.write(str(a) + " " + str((date1-prev).total_seconds()/len(line)))
					writer.writerow(row+[str((date1-prev).total_seconds()/len(line))])
				prev= date1
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'_final'+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'_final'+'.csv')	


f = sys.argv
if __name__ == "__main__":
	writing_speed_android(str(f[1]),str(f[2]),str(f[3]),0)
