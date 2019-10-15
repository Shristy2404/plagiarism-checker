# importing required modules
import csv
import random
import math
import operator
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# global sets
training_set=[]
test_set=[]

# values to replace the string attributes in the data
values={}
values[0]={"usual":1, "pretentious":2, "great_pret":3}
values[1]={"proper":1, "less_proper":2, "improper":3, "critical":4, "very_crit":5}
values[2]={"complete":1, "completed":2, "incomplete":3, "foster":4}
values[3]={"1":1, "2":2, "3":3, "more":4}
values[4]={"convenient":1, "less_conv":2, "critical":3}
values[5]={"convenient":1, "inconv":2}
values[6]={"nonprob":1, "slightly_prob":2, "problematic":3}
values[7]={"recommended":1, "priority":2, "not_recom":3}

# function to divide the dataset into test set and training set randomly 
def divide_data(fraction, dataset):
	for i in range(len(dataset)-1):
		if random.random() < fraction:
			training_set.append(dataset[i])
		else:
			test_set.append(dataset[i])

# function to the load the .data file into a list 
def load_data(file):
	with open(file,'rt') as csvfile:
		lines=csv.reader(csvfile)

		# list of lists 
		dataset=list(lines)	
		for i in range(len(dataset)-1):
			for j in range(8):
				dataset[i][j]=values[j][dataset[i][j]]
	return dataset

# computes Euclidean distance between two data instances
def euclidean_dist(data1, data2):
	dist=0
	for i in range(8):
		dist+=pow((data1[i]-data2[i]),2)
	dist=math.sqrt(dist)
	return dist

# Returns the class label occuring maximum number of times in k-nearest neighbours 
def get_max_votes(neighbours):

	votes={}
	for i in range(len(neighbours)):
		res=neighbours[i][-1]
		if res in votes:
			votes[res]+=1
		else:
			votes[res]=1
	votes=sorted(votes.items(),key=operator.itemgetter(1), reverse=True)
	return votes[0][0]

# returns list of k-nearest neighbours from the training set for a given instance of test set
def neighbours_list(test,k):
	distances=[]

	for i in range(len(training_set)):
		dist = euclidean_dist(test,training_set[i])
		distances.append((training_set[i],dist))
	distances.sort(key=operator.itemgetter(1))
	neighbours=[]
	for i in range(k):
		neighbours.append(distances[i][0])
	return neighbours

# calculates accuracy of the calculated predictions based on the actual values 
def accuracy(predicted):
	right=0
	for i in range(len(test_set)):
		if(test_set[i][-1] == predicted[i]):
			right+=1
	if(len(test_set)>0):			
		acc = (right/float(len(test_set)))*100.0
	return acc
	

if __name__ == '__main__':
	dataset = load_data('nursery.data')
	acc_sum=0.0 
	predicted=[]
	actual=[]
	k=5
	# 5 iterations for reducing randomness introduced by splitting 
	for x in range(1):
		initial = time.time()
		print ("Iteration :" + str(x))
		# using standard 0.67 dividing ratio
		divide_data(0.67, dataset)
		for i in range(len(test_set)):
			neighbours=neighbours_list(test_set[i],k)
			res = get_max_votes(neighbours)
			predicted.append(res)
			actual.append(test_set[i][-1])
			print('> predicted=' + res + ', actual=' + test_set[i][-1])
		acc = accuracy(predicted)
		print ('Accuracy for iteration ' + str(x) + " is " + str(acc) + "%")
		acc_sum+=acc
		print ('Time taken : ' + str(time.time()-initial))
	avg_acc = acc_sum/5.0
	a=accuracy_score(predicted, actual)
	c=confusion_matrix(predicted, actual)
	cr=classification_report(actual, predicted)
	print (a)
	print (c)
	print (cr)
	print ('Average accuracy over 5 iterations :' + str(avg_acc) + "%")