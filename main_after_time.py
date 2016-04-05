#!/usr/bin/python
# to incorporate the writing speed feature of the indivuduals
import csv
from process_time_chat import time_chat_android
from process_time_chat1 import time_chat_windows
from writing_speed import writing_speed_android
from writing_speed1 import writing_speed_windows
import sys
import os
#take file inp

path = './chats_process'

def main_after_time():

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
		print first,second
		
		if not os.path.exists('./chats_process/'+str(first)+'_'+str(second)+'/'):
			os.makedirs('./chats_process/'+first+'_'+second+'/')	
	
		writing_speed_android('./chats_process/'+str(first)+'_'+str(second)+'/'+first+'.txt',str(first),str(second),1)
		writing_speed_android('./chats_process/'+str(first)+'_'+str(second)+'/'+second+'.txt',str(first),str(second),2)


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
					if l.find(":")!=-1:
						a=l.split(":")
						second = a[3]
				
						print "in loop"
						break
			break	
			f.close()
		print first,second
	
		if not os.path.exists('./chats_process/'+str(first)+'_'+str(second)+'/'):
			os.makedirs('./chats_process/'+first+'_'+second+'/')	

		writing_speed_windows('./chats_process/'+str(first)+'_'+str(second)+'/'+first+'.txt',str(first),str(second),1)
		writing_speed_windows('./chats_process/'+str(first)+'_'+str(second)+'/'+second+'.txt',str(first),str(second),2)

