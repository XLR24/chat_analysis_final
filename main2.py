#!/usr/bin/python
#Things to extract
import csv
from filter_chat import extract
from average_word import average_word
from word_length import word_length
from upper_low_ratio import ratio
from msg_length import msg_length
from punctuation_usage import punctuation
from stopword_freq import stopword_usage
from smileys import smiley_count
from acro_line import acro_line
import sys
import os
#take file inp

def compute_features():

	t=open("./package/training_set_3.csv", 'w')
	c = csv.writer(t)
	c.writerow(["serial no"])

	i=1
	while i<10:
		c.writerow([i])
		i=i+1

	t.close()


	#two files 1_chat and 2_chat are created
	#first feature to extract ////"""average words per line""""
	average_word("./package/test_final.txt",3)#creates a text file named number_word_1 which contains number of word in each line.
	#print "average_word_feature extracted"

	#second feature """word length per line"''
	word_length("./package/test_final.txt",3)#creates a text file named word_length_1 which contains number of word in each line.
	#print "word_length_feature extracted"

	#third feature ""uppercase_lowrcase ratio""""
	ratio("./package/test_final.txt",3)#creates a text file named word_length_1 which contains number of word in each line.
	#print "ratio_feature extracted"

	#fourth feature ""time_chat , time to give that particular reply"""


	#sixth feature """smiley count """every 10 line feature usage
	smiley_count("./package/test_final.txt",3)
	#print "smiley count extracted"

	#fifth feature """stopwaord usagee in a line""""every 10 line feature
	stopword_usage("./package/test_final.txt",3)
	#print "stopword feature extracted"

	#seventh feature """punchuation usage""""
	punctuation("./package/test_final.txt",3)
	#print "punctuation_feature extracted"


	#eighth feature """message length ""
	msg_length("./package/test_final.txt",3)
	#print "msg_length_feature extracted"

	#ninth feature """acronym count """
	acro_line("./package/test_final.txt",3)
	#print "acro_line extracted"

	#tenth feature """how much he has used his common words in these line"""




