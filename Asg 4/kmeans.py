import csv
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import matplotlib.gridspec as gridspec


def sse(previous,centres):
	len1 = len(previous)
	total =0
	# print previous.shapee
	for i in range(0,len1):
		dis = np.square((previous[i][0]- centres[i][0]))
		dis+= np.square((previous[i][1]- centres[i][1]))
		dis=np.sqrt(dis)
		total+=dis
	return total

# input data
def kmeans(dataset,k,iterations):
	print "Dataset: ", dataset
	print "Iterations: ",iterations
	data = [] 
	with open(dataset, 'r') as file:
	    for row in file:
	        a, b = row.split()
	        l = [float(a),float(b)]
	        data.append(l)
	data = np.array(data)
	r=5
	error = 9999
	# Number of training data
	n = data.shape[0]
	# Number of features in the data
	c = data.shape[1]

	fig, axs = plt.subplots(5,3)
	fig.suptitle(dataset)
	for j in range(0,r):

		
		print "r=",j
		# intialize centers randomly
		mean = np.mean(data, axis = 0)
		std = np.std(data, axis = 0)
		centres = np.random.randn(k,c)
		
		axs[j][0].scatter(data[:,0], data[:,1], s=7)
		axs[j][0].scatter(centres[:,0], centres[:,1], marker='*', c='r', s=150)
		if(j==0):
			axs[j][0].set_title("Before clustering")
		
		listError=[]
		numIteration = 0
		error = 999999
		while(numIteration<100):
			numIteration+=1
			classes=[]
			for i in range(0,k):
				classes.append([])

			# classify data to clusters
			for sample in data:
				# find distance from every centre
				distances = [np.linalg.norm(sample-centre) for centre in centres ]
				# cal min dist
				minDistance  = min(distances)
				# get its index
				indexMin = distances.index(minDistance)
				# add to that clusters
				classes[indexMin].append(sample)
			
			# store previous centers
			previous = deepcopy(centres)
			# recalculate centres  by taking mean of the clusters

			for i in range(0,k):
				centres[i]= np.mean(classes[i],axis=0)


			# Error calculations
			error = sse(previous,centres)
			
			if(numIteration==1):
				print "Error at first iteration {0:0.3f} ".format(error)
			
			# dince the above function gives square root value
			# error = error*error
			listError.append(error)
			# print error
		# print numIteration
		print "Error at last iteration  {0:0.3f} ".format(error)

		# errors and iterations plot for different r
		iterations=[h for h in range(0,numIteration)]
		# print iterations
		if(j==0):
			axs[j][2].set_title("Error vs Iterations")
		axs[j][2].plot(iterations,listError,'r')
		# axes[j].set_title("\nErros Vs Iterations\n")
		axs[j][2].set_xlabel('Iterations')
		axs[j][2].set_ylabel('Error')

		# print classes[0][0][0]
		colors = np.array([x for x in 'bgrcmyk'])

		# clusters 
		if(j==0):
			axs[j][1].set_title("After clustering")
		
		for o in range(0,k):
			twodarray = classes[o]
			twodarray = np.array(twodarray)
			axs[j][1].scatter(twodarray[:,0], twodarray[:,1],c=colors[o], s=7)
		axs[j][1].scatter(centres[:,0], centres[:,1], marker='*', c='k', s=150)
	plt.show()
	return classes



def main_1(dataset,k,iterations):
	classes = kmeans(dataset,k,iterations)
	return classes

if __name__ == '__main__':
	main_1('Dataset_1.txt',2,100)
	main_1('Dataset_2.txt',3,100)


# error and iterations
# ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
# ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
# ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
# ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
# ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

# AX = gridspec.GridSpec(2,5)
# ax1 = plt.subplot(AX[0,0])
# ax2  = plt.subplot(AX[0,2])
# ax3 = plt.subplot(AX[0,4])
# ax4 = plt.subplot(AX[1,1])
# ax5 = plt.subplot(AX[1,3])
# axes = [ax1,ax2,ax3,ax4,ax5]
