import os
from datetime import datetime
from datetime import timedelta
f = open('time.txt', 'r')
z3 = open('chat.txt', 'r')
prev=0
date1=0

z = open('turn_duration.txt', 'w')
z1 = open('response_time.txt', 'w')
z4 = open('chat_turn.txt', 'w')
z5 = open('chat_session.txt', 'w')

for line,line1 in zip(f,z3):
	
	a = line.index(":::")
	p=line[:a]
	line = line[a:]
	
	date_session=datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")



	if prev==0:
		date1 = datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")
		prev=1
		prev_turn=p
		prev_date_session=date_session

	if prev==1 and p==prev_turn:
		date3 = datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")
	else:
		date2 = datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")
		delta= date3-date1

		delta1 = date2-date3

		z1.write(str(delta1) + "\n")
		#print delta
		prev=1
		z4.write("\n------------------\n")
		z.write( str(prev_turn)+  " " + str(delta) + "\n")
		prev_turn=p
		date1 = datetime.strptime(line,"::: %m/%d/%Y, %I:%M %p\n")
		date3=date1

	z4.write(line1)
	

	delta_session = date_session - prev_date_session

	#t = datetime.strptime("02:00:00","%H:%M:%S")

	if(delta_session > timedelta(hours=2)):
		prev_date_session=date_session
		z5.write("\n--------\n\n")

	z5.write(line1)


		
		


	
