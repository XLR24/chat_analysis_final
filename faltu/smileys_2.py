from collections import defaultdict
import codecs

s = open('smiley_2.txt','wb')

with codecs.open('2_chat.txt', encoding='utf-8') as f:
    lines = f.read().splitlines()

cnt = 0
a=0

for line in lines:
    a+=1
    words = line.split()
    for word in words: 
        for letter in word:
            st=letter.encode('utf-8')
            if st >= '\xF0\x9F\x98\x81' and st <='\xF0\x9F\x99\x8F':
                cnt+=1
    if (a%10) == 0: 
        s.write(str(a) + " -> " + str(cnt) )
        s.write("\n")
        cnt=0


                
