import csv
import sys
import os

path = './chats_process'
for filename in os.listdir(path):
	
	filename=filename.split("_")
	first=filename[0]
	second=filename[1]

	#print first
	#print second

	#print "b"
	with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_1_final.csv','r') as f:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_2_final.csv','r') as ff:
			with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'train.csv','w') as f1:
				f.next() # skip header line
				ff.next()
				#print "a"
				for line in f:
					line = line.replace('\n', '').replace('\r', '')
					line=line+',1\n'

					lines=line.split(',')
					a=0
					linet=''
					for word in lines:
						if a==0:
							a=a+1
							continue
						linet=linet+','+word
		
					
					linet = linet[1:]
						
					print linet
					f1.write(linet)

				for line in ff:
					line = line.replace('\n', '').replace('\r', '')
					line=line+',0\n'
					lines=line.split(',')
					a=0
					linet=''
					for word in lines:
						if a==0:
							a=a+1
							continue
						linet=linet+','+word
		
					
					linet = linet[1:]
						
					#print line
					f1.write(linet)
	
			f1.close()
		ff.close()
	f.close()

#def combine_train():

	
