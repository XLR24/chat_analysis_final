#filtering the chat
import os
import sys
f = sys.argv
if __name__ == "__main__":
	extract(str(f[1]))
def extract(fn):
	f = open(str(fn), 'r')
	x=0
	
	for line in f:
		#print line
		a = line.split(":")
		
		first = a[1]

		b=first.split("-")

		first=b[1]


		f = open(str(fn), 'r')
		for l in f:
			if l.find(first)==-1:
				a=l.split(":")
				second = a[1]
				b=second.split("-")

				second=b[1]

				print "in loop"
				break
		break	
	print first,second
	
	f = open(str(fn), 'r')
	a=0
	for line in f:
		if line.find(first)!=-1:
			
			s = open('./package/1_chat.txt','a')
			index_1 = line.index(first)

			u = line[index_1:]
			m = u.index(":")
			t = u[m:]
		
			a=0
			s.write(t)
		elif line.find(second)!=-1:
			p = open('./package/2_chat.txt','a')
			index_2 = line.index(second)
			u = line[index_2:]
			m = u.index(":")
			t = u[m:]
			
			a=1
			p.write(t)
		else:
			if a==0:
				s = open('./package/1_chat.txt','a')
				s.write(line)
			else:
				p = open('./package/2_chat.txt','a')
				p.write(line)



		
