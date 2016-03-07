import os
import sys
import csv
f = sys.argv

def media_url(fi,first,second,num):
    with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv','r') as csvinput:
		with open('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', 'w') as csvoutput:
			writer = csv.writer(csvoutput)
			a=0
			for row in csv.reader(csvinput):
				writer.writerow(row+["media_url"])
				break
			f = open(fi, 'r')
			for row,line in zip(csv.reader(csvinput),f):	
				s = open('./chats_process/'+str(first)+'_'+str(second)+'/'+'media_url_'+str(num)+'.txt','a')
				a=0

				count_media = 0
				count_url = 0

				
				if line.find("<Media omitted>")>=1 or line.find("<image omitted>")>=1:
					count_media+=1
				if line.find("http:")>=1 or line.find("https:")>=1 :
					count_url+=1		

			
				s.write(str(a) + " -> " + str(count_media) + ", " + str(count_url))
				writer.writerow(row+[count_media+count_url])
				s.write("\n")
				count_media=0
				count_url=0
				a=a+1


				s.close()
			f.close()
			
			os.rename('./chats_process/'+str(first)+'_'+str(second)+'/'+'training_'+str(num)+'.csv', './chats_process/'+str(first)+'_'+str(second)+'/'+'training_set_'+str(num)+'.csv')

if __name__ == "__main__":
	media_url(str(f[1]),str(f[2]),str(f[3]),0)
