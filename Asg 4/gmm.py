import csv
import numpy as np
from matplotlib import pyplot as plt
import pandas
from copy import deepcopy
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal
from kmeans import main_1

def initialiseRandom(data,k):
	mean = np.random.rand(k,2)
	data = data.T
	covar = np.identity(2)
	covar[0][1]= 1e-6
	covar[1][0]= 1e-6
	covariance =  [covar for i in range(0,k)]
	covariance = np.array(covariance)
	return mean,covariance



def getMeanAndCovar(classes,k,n):
	mean = np.zeros((k,2))
	covariance=np.zeros((k,2,2))
	# intialization mean,cov from k means
	for i in range(0,k):
		# mean
		m=[0,0]
		classes[i]=np.array(classes[i])
		m[0]=np.sum(classes[i][:,0])/n
		m[1]=np.sum(classes[i][:,1])/n
		# mean of cluster
		mean[i] = m

		temp = np.cov(classes[i].T)
		covariance[i]= temp		

	# print 
	return mean,covariance





def gmm(dataset,data,k,iterations,mean,covariance,flag): 

	# mean.shape - k*2 covariance.shape = k*2*2
	
	n = data.shape[0]
	r= np.zeros((n,k),dtype=float)
	
	m_c = np.zeros((k))
	mu_c = np.zeros((k,2))
	cov_c =np.zeros((k,2,2))

	pi_c = np.zeros((k)) 
	
	mu_c = deepcopy(mean)
	cov_c = deepcopy(covariance)
	# print float(1/k)
	# initially equal weights for all gaussians
	pi_c = [float(1)/float(k) for i in range(0,k)]
	logLikelihood=[]
	# print mu_c,cov_c

	for o in range(0,iterations):
	
# Expectation step 
	
		# gaussiansfig, axs = plt.subplots(5,3)
		# print bivariate_normal(data[:,0],data[:,1],covariance[j][:,0],covariance[j][:,1],mean[j][0],mean[j][1]).shape
		for i in range(0,n):
			for j in range(0,k):
				# probability that x[i] belongs to gaussian j
				r[i][j] = pi_c[j]*multivariate_normal.pdf(data[i], mean=mu_c[j], cov=cov_c[j])


		# calculate weights ric =  probability that i belongs to c
		for i in range(len(r)):
			# print np.sum(r[i])
			r[i] = r[i]/(np.sum(r[i]))


		# maximization step
		# For each cluster c, calculate the m_c and add it to the list m_c
		# m_c[c] = fraction of points that belong to cluster c.
		
		for c in range(0,k):
		    m = np.sum(r[:,c])
		    m_c[c] = m


		"""calculate pi_c"""
		# For each cluster c, calculate the fraction of points pi_c which belongs to cluster c
		# print np.sum(m_c),n
		for i in range(0,k):
		    pi_c[i] = (m_c[i]/n) #

		"""calculate mu_c"""
		for i in range(0,k):
			mu = np.zeros((2),dtype=float)
			mu[0] = np.sum(r[:,i]*data[:,0])/m_c[i]
			mu[1] = np.sum(r[:,i]*data[:,1])/m_c[i]
			mu_c[i] = mu

		mu_c = np.array(mu_c)

		# 5*2 mean vectors for 5 clusters
		# print mu_c.shape

		# """calculate var_c"""
		temp = np.zeros((2,2))
		for i in range(0,k):
			l1 = r[:,i]*(data[:,0] - mu_c[i][0])
			l2 = r[:,i]*(data[:,1] - mu_c[i][1])
			l=[l1,l2]
			l = np.array(l)
			l = l.reshape(2,n)
			temp = np.dot(l,l.T)
			cov_c[i]= temp/m_c[i]
		# print cov_c

		# log likelihood
		totalOverall=0
		for i in range(0,n):
			total=0
			for j in range(0,k):
				total+=pi_c[j]*multivariate_normal.pdf(data[i], mean=mu_c[j], cov=cov_c[j])
			# print total
			totalOverall+= np.log(total)
		logLikelihood.append(totalOverall)
		if (o%10==0):
			print "Log likelihood after ", o," iterations ", totalOverall
		# print totalOverall

# plot contours
	fig, axs = plt.subplots(1, 3, constrained_layout=True)
	if flag:
		fig.suptitle(dataset + " k means Initialisation")
	else:
		fig.suptitle(dataset + " random Initialisation")
		
	axs[0].scatter(data[:,0],data[:,1])
	x,y = np.meshgrid(np.sort(data[:,0]),np.sort(data[:,1]))
	XY = np.array([x.flatten(),y.flatten()]).T
	
	for m,c in zip(mu_c,cov_c):
		multi_normal = multivariate_normal(mean=m,cov=c)
		axs[0].contour(np.sort(data[:,0]),np.sort(data[:,1]),multi_normal.pdf(XY).reshape(len(data),len(data)),8,colors='black',alpha=0.8,)
		axs[0].scatter(m[0],m[1],c='grey',zorder=10,s=100)
		axs[0].set_title('Contours')
	

# plot clusters
	clusterLabel=[];
	for h in range(0,n):
		c = np.argmax(r[h,:])
		clusterLabel.append(c);

	axs[1].scatter(data[:,0],data[:,1],c=clusterLabel)
	axs[1].set_title("Clusters")
	
	# print logLikelihood	

# plot log likelhood
	
	numIterations = [i for i in range(0,iterations)]
	axs[2].plot(numIterations,logLikelihood)
	axs[2].set_title('Iterations Vs logLikelihood')
	
	plt.show()

	# axs.scatter(data[:,0], data[:,1], s=7)
	# axs.scatter(centres[:,0], centres[:,1], marker='*', c='r', s=150)
	return logLikelihood,multivariate_normal.pdf(data[i], mean=mu_c[j], cov=cov_c[j])

def main(dataset,k,iterations,flag):
	# input data 
	print "Dataset: ", dataset
	print "Iterations: ",iterations
	data = []
	with open(dataset, 'r') as file:
	    for row in file:
	        a, b = row.split()
	        l = [float(a),float(b)]
	        data.append(l)
	data = np.array(data)
	n = data.shape[0]
	# random initialisation
	if(flag==0):
		mean,covariance  = initialiseRandom(data,k)
		gmm(dataset,data,k,iterations,mean,covariance,0)
	else:
		classes = getClasses()
		mean,covariance = getMeanAndCovar(classes,k,n)
		gmm(dataset,data,k,iterations,mean,covariance,1)
		

	#using kmeans 

def getClasses():
	return classes


if __name__ == '__main__':


	# can execute one at a time.. randome init and kmeans init
	# or both should also work fine,


	# intitalization random
	print "Random Initialisation\n"
	main('Dataset_1.txt',2,100,0)
	main('Dataset_2.txt',3,100,0)

	# K means initialisation

	print "\nRunning kmeans on Dataset 1 to get mean and covariance of data for initialisation\n"
	# dataset1
	# get classifications from kmeans
	classes1 = main_1('Dataset_1.txt',2,100)
	classes = classes1
	# call gmm 
	print "\nGMM on Dataset 1"
	main('Dataset_1.txt',2,100,1)


	# dataset_2
	print "\nRunning kmeans on Dataset 2 to get mean and covariance of data for initialisation\n"

	classes2 = main_1('Dataset_2.txt',3,100)	
	classes = classes2
	# call gmm
	print "\nGMM on Dataset 2"
	main('Dataset_2.txt',3,100,1)


	# calculate means and covariance use that for intialisation




	

	
