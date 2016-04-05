#filtering the chat
import os
import sys
#this file seperates the chat of two people taken one by one from "raw" folder
"""
Raw folder contains two folders "android" and "windows" which consist the chat txt emailed from whatsapp.The main motive to create two seperate folder is bcz android and windows are having different formats in term of date and time,so extraction is also different.
"""
def process():
	
	path = './raw/android/'
	#for android chats only
	for filename in os.listdir(path):
		f = open('./raw/android/'+filename, 'r')
		x=0
		
		for line in f:
			#print line
			a = line.split(":")
			
			first = a[1]

			b=first.split("-")

			first=b[1]
			#get the first user name

			f = open('./raw/android/'+filename, 'r')
			for l in f:
				if l.find(first)==-1:
					a=l.split(":")
					#print l
					if len(a)>1:
						second = a[1]
						b=second.split("-")
						if(len(b)>1):#for catching errors if there is no "-" in line
							second=b[1]

							
							break#break if u get the name of second user
			break	
		print first,second
		
		if not os.path.exists('./chats_process/'+str(first)+'_'+str(second)+'/'):#make a folder of name first_second user if it doesn't exist
			os.makedirs('./chats_process/'+first+'_'+second+'/')		
		# Now we seperate out the chat of two different user from the overall conversation
		f = open('./raw/android/'+filename, 'r')
		a=0
		for line in f:
			if line.find(first)!=-1:
			
				s = open('./chats_process/'+first+'_'+second+'/'+first+'.txt','a')
				index_1 = line.index(first)
				
				u = line[index_1:]
				if u.find(":")!=-1:
					m = u.index(":")
					t = u[m:]#splitting the timestamp and text
			
					a=0
					s.write(t)#writing only text to a new file including ":" as it indicates the start of a new line
			elif line.find(second)!=-1:#if the line is written by second user,copy it in another txt file of name second user.
				s = open('./chats_process/'+first+'_'+second+'/'+second+'.txt','a')
				index_2 = line.index(second)
				u = line[index_2:]
				if u.find(":")!=-1:
					m = u.index(":")
					t = u[m:]
				
					a=1
					s.write(t)
			else:
				if a==0:#if no user description present in line .it mens it is to be copied in the previous user txt
					s = open('./chats_process/'+first+'_'+second+'/'+first+'.txt','a')
					s.write(line)
				else:
					s = open('./chats_process/'+first+'_'+second+'/'+second+'.txt','a')
					s.write(line)
	
	#for chat obtained from windows phone,there is a change of format
	# All the things that are done in case of android is repeted ,just some of the extraction varies due to format of chat.
	path = './raw/windows'
	for filename in os.listdir(path):
		f = open('./raw/windows/'+filename, 'r')
		x=0
		
		for line in f:
			#print line
			a = line.split(":")
			
			first = a[3]



			f = open('./raw/windows/'+filename, 'r')
			for l in f:
				if l.find(first)==-1:
					a=l.split(":")
					second = a[3]
					
					print "in loop"
					break
			break	
			f.close()
		print first,second
		
		if not os.path.exists('./chats_process/'+str(first)+'_'+str(second)+'/'):
			os.makedirs('./chats_process/'+first+'_'+second+'/')		

		f = open('./raw/windows/'+filename, 'r')
		a=0
		for line in f:
			if line.find(first)!=-1:
			
				s = open('./chats_process/'+first+'_'+second+'/'+first+'.txt','a')
				index_1 = line.index(first)

				u = line[index_1:]
				if u.find(":")!=-1:
					m = u.index(":")
					t = u[m:]
			
					a=0
					s.write(t)
			elif line.find(second)!=-1:
				s = open('./chats_process/'+first+'_'+second+'/'+second+'.txt','a')
				index_2 = line.index(second)
				u = line[index_2:]
				if u.find(":")!=-1:
					m = u.index(":")
					t = u[m:]
					a=1
					s.write(t)
			else:
				if a==0:
					s = open('./chats_process/'+first+'_'+second+'/'+first+'.txt','a')
					s.write(line)
				else:
					s = open('./chats_process/'+first+'_'+second+'/'+second+'.txt','a')
					s.write(line)

	
if __name__ == "__main__":
	process()
		
