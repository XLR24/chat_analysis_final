import os
f = open('1_chat.txt', 'r')

count = 0
emoticons = set(range(int('1f600',16), int('1f650', 16)))
for row in f:
    for char in row:
        if ord(char) in emoticons:
            count += 1
	print "%d emoticons found" % count