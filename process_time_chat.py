#for obtaining response time of chat in android files
import os
import sys
from datetime import datetime
from datetime import timedelta
import time
import csv

def time_chat_android():

	path = './raw/android/'
	for filename in os.listdir(path):
		f = open('./raw/android/'+filename, 'r')
		x=0
		
		for line in f:
			#print line
			a = line.split(":")
			
			first = a[1]

			b=first.split("-")

			first=b[1]


			f = open('./raw/android/'+filename, 'r')
			for l in f:
				if l.find(first)==-1:
					a=l.split(":")
					
					if len(a)>1:
						second = a[1]
						b=second.split("-")
						if(len(b)>1):
							second=b[1]

							#print "in loop"
							break
			break	
		
		f = open('./raw/android/'+filename, 'r')
		a=1 #first person speaking
		s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'time.txt','w')
		s.close()
		for line in f:
			s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'time.txt','a')
			line.replace("," , " ::: ")
			if line.find(first)!=-1:
			
				if line.find(" - ")!=-1:
					index_1 = line.index(" - ") #truncate by #

					t = line[:index_1] # put date and time in a file
					#print t
					a=1
					s.write(str(a) + " ::: " +t)

					s.write("\n")
					s.close()
			elif line.find(second)!=-1:
				if line.find(" - ")!=-1:
					index_2 = line.index(" - ")
					t = line[:index_2]
					a=2 # second person speaking
					s.write(str(a) + " ::: " +t)
					s.write("\n")
					s.close()
		#f.close()

		f = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'time.txt', 'r')
		z3 = open('./raw/android/'+filename, 'r')
		prev=0
		date1=0
		delta1=0
		z = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'turn_duration.txt', 'w')
		z1 = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'response_time.txt', 'w')
		z4 = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'chat_turn.txt', 'w')
		z5 = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'chat_session.txt', 'w')

		#x = time.strptime('00:01:00,000'.split(',')[0],'%H:%M:%S')
		#datetime.timedelta(days=x.tm_mdayhours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()

		for line,line1 in zip(f,z3):
			
			a = line.index(":::")
			p=line[:a]
			line = line[a:]
			correctDate=False
			#if date is in 24 hrs format
			#print line
			try:
			    date_session=datetime.strptime(line,"::: %d/%m/%Y, %H:%M\n")
			    correctDate=True
			except ValueError:
				#print "dsj"
				correctDate=False
				
			#convert to 12 hrs format
			#print correctDate	
			if correctDate==True:
				
				date_session.strftime("::: %m/%d/%Y, %I:%M %p\n")
			else:
				
				date_session=datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")
				
			if prev==0:
				if correctDate:
					date1 = datetime.strptime(line,"::: %d/%m/%Y, %H:%M\n")
					date3 = datetime.strptime(line,"::: %d/%m/%Y, %H:%M\n")	
					date1.strftime("::: %m/%d/%Y, %I:%M %p\n")
					date3.strftime("::: %m/%d/%Y, %I:%M %p\n")
				else:
					date1 = datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")
					date3 = datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")				
				prev=1
				prev_turn=p
				prev_date_session=date_session
				z1.write( str(prev_turn)+  " " + str(delta1) + "\n")

			elif prev==1 and p==prev_turn:
				if correctDate:
					date3 = datetime.strptime(line,"::: %d/%m/%Y, %H:%M\n")
					date3.strftime("::: %m/%d/%Y, %I:%M %p\n")
				else: 
					date3 = datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")
				prev_turn=p
				z1.write( str(prev_turn)+  " " + str(delta1) + "\n")
			else:
				if correctDate:
					date2 = datetime.strptime(line,"::: %d/%m/%Y, %H:%M\n")
					date2.strftime("::: %m/%d/%Y, %I:%M %p\n")
				else:
					date2 = datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")
				
				if date3>date1:
					delta= date3-date1
					delta=delta.total_seconds()
				else:
					delta= date1-date3
					delta=delta.total_seconds()

				if date3 > date2:
					delta1 = date3-date2
					delta1=delta1.total_seconds()
				else:
					delta1 = date2-date3
					delta1=delta1.total_seconds()

				z1.write( str(prev_turn)+  " " + str(delta1) + "\n")
				#print delta
				prev=1
				z4.write("\n------------------\n")
				z.write( str(prev_turn)+  " " + str(delta) + "\n")
				prev_turn=p
				if correctDate:
					date1 = datetime.strptime(line,"::: %d/%m/%Y, %H:%M\n")
					date1.strftime("::: %m/%d/%Y, %I:%M %p\n")
				else:
					date1 = datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")
				date3=date1

			z4.write(line1)
			

			delta_session = date_session - prev_date_session

			#t = datetime.strptime("02:00:00","%H:%M:%S")

			if(delta_session > timedelta(hours=2)):
				prev_date_session=date_session
				z5.write("\n--------\n\n")

			z5.write(line1)	
		
		f.close()
		z3.close()
		z.close()
		z1.close()
		z4.close()
		z5.close()
		
		
		
		z1 = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'response_time.txt', 'r')
		z2 = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'response_time_username1.txt', 'w')

		
		#t=open('./chats_process/'+str(first)+'_'+str(second)+'/'+'train.csv', 'w')
		
		for line in z1:
			
			line = line.split(" ")
			
			fir=line[0]
			
			seconds = line[2]
			#seconds = seconds.replace('\n', '').replace('\r', '')

			#print seconds
			
			if str(fir)=='1':
				z2.write(seconds)
		
		z2.close()
		z1.close()

		
		z2 = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'response_time_username2.txt', 'w')
		z1 = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'response_time.txt', 'r')

		for line in z1:
			
			line = line.split(" ")
			
			fir=line[0]
	
			seconds = line[2]

			if fir=='2':
				z2.write(seconds)
		
		z1.close()
		z2.close()
		
		
		z2 = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'response_time_username1.txt', 'r')
		
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_1.csv','r') as f:
			with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_1_final.csv', 'w') as ff:
				writer = csv.writer(ff)
				
				for row in csv.reader(f):
					writer.writerow(row+["response_time"])
					break
				
				for row,line in zip(csv.reader(f),z2):
					line = line.replace('\n', '').replace('\r', '')
					writer.writerow(row+[line])
		
		f.close()
		ff.close()
		z2.close()

		z2 = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'response_time_username2.txt', 'r')
		
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_2.csv','r') as f:
			with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_2_final.csv', 'w') as ff:
				writer = csv.writer(ff)
				
				for row in csv.reader(f):
					writer.writerow(row+["response_time"])
					break

				for row,line in zip(csv.reader(f),z2):
					line = line.replace('\n', '').replace('\r', '')
					writer.writerow(row+[line])
		
		f.close()
		ff.close()
		z2.close()


if __name__ == "__main__":
	time_chat_android()
		
