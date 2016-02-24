import csv
import random
import math
import csv
from main2 import compute_features
 
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset
 
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
    
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
 
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
 
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries
    
def calculateProbability(x, mean, stdev):
	if stdev==0:
		stdev=0.00000001
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
    
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
 
def main():
	filename = './package/train_1.csv'
	splitRatio = 1
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	#print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	#type(summaries)
	#print('Summary by class value: {0}').format(summaries)
	
	testVar = raw_input("Write a message.\n")
	testVar = ": "+ testVar
	testVar	= "\n"+ testVar
	with open('./package/test_final.txt', 'w') as the_file:
		the_file.write(testVar)

	#print "hey"
	compute_features()

	lines = csv.reader(open('./package/training_set_3.csv', "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		if i==1:
			dataset[i] = [float(x) for x in dataset[i]]
	
	dataset.pop(0)	

	print 'These are the feature values that we have calculated for you input\n'

	datas = dataset[0]
	print datas
	
	for idx, a in enumerate(datas):
		if datas[idx]==0.0:
			datas[idx]=1
    	

	#print datas
	
	print '\nHere is the result\n\n'

	

	probabilities = calculateClassProbabilities(summaries, datas)
	#probabilities["aa"] = 12
	#print probabilities

	proba = probabilities[0.0]
	probb = probabilities[1.0]

	proba = proba * .5
	probb = probb * .5

	user1 = proba/(proba+probb)
	user2 = probb/(proba+probb)

	if(user1 > user2):
		print "You are imposter"
	else:
		print "You are legitimate"


	#type(probabilities)
	'''proba = probabilities['0.0']
	probb = probabilities['1.0']

	print proba
	print "\n"
	print probb
	'''
	#print('Probabilities for each class: {0}').format(probabilities)

    
main()