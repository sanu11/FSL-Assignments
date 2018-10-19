from loaddata import mnist
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
global m,et,wt_matrix
from numpy import linalg,diag,sqrt

#--------------Create database---------------------------------


def whiten(X,fudge=1E-3):

   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = np.dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d, V = np.linalg.eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1. / np.sqrt(d+fudge))

   # whitening matrix
   W = np.dot(np.dot(V, D), V.T)

   # multiply by the whitening matrix
   X_white = np.dot(X, W)

   return X_white


def Reduce(data):
	global m,et,wt_matrix
	
	ImageVector=np.asarray(data)												 #convert list to np.array 
	# print ImageVector.shape
	#----------------Eigen_faces------------------------------------

	m=np.mean(ImageVector,axis=1)								 #mean about columns (axis=1 gives mean about column)

	A=ImageVector-m[:,None]    							         #Difference matrix(Unique features)
	At=A.T 														 #Transpose of Difference matrix				
	covar=A.dot(At)												 #covariance = A transpose into A

	[U,S,V] = linalg.svd(covar);

	# print U.shape,S.shape,A.shape
	xRot = U.T.dot(A)     
	xTilde = U[:,1:101].T.dot(A)
	return xTilde,U,S,m


def plotData(xTilde,U,S):
	size = S.shape[0]
	sumLambda=0
	sumLambda2=0	


	kEigenVec = U[:,1:101]
	# print kEigenVec.shape 
	for i in range(0,size):
		sumLambda+= S[i]
		if(i<=100):
			sumLambda2+= S[i]


	print "Variance Retained after dimensionality reduction: ",(sumLambda2/sumLambda)*100
	
# ---------------PCA COV------------------------
	covPCA = np.cov(xTilde)
	#reduced dimenions to 100 
	#images in the form of eigen vectors
	#thus 100*1000 dimension 
	print "PCA reduced data: ",xTilde.shape


# -----------PCA digits---------------------
	# Plot 10 digits
	fig, ax = plt.subplots(2, 5, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
	for i in range(5):
		# print 100*i
		ax[0, i].imshow(xTilde[:,100*i].reshape(10, 10), cmap='jet')

	for i in range(5):
	   	 ax[1, i].imshow(xTilde[:,100*i+500].reshape(10, 10), cmap='jet')
	
	fig.suptitle("PCA digits")
	plt.show()


# -------------------___PCA whiten cov-------------------
	epsilon = 1E-3;


	PCAwhite = whiten(xTilde)
	covPCAwhite = np.cov(PCAwhite)
	print "PCA white: ",PCAwhite.shape


# --------------ZCA whiten cov----------------------
	
	ZCAWhite = kEigenVec.dot(PCAwhite)	
	covZCA = np.cov(ZCAWhite)

	print "ZCA white: ",ZCAWhite.shape


# -------------plot all variances------------
	
	fig, (arr1,arr2,arr3) = plt.subplots(1, 3)
	fig.suptitle("Covariances", fontsize=16)

	arr1.matshow(covPCA)
	arr1.set_title("\nPCA Covariance\n")
	
	arr2.matshow(covPCAwhite)
	arr2.set_title("PCA White Covariance\n")
	
	arr3.matshow(covZCA)
	arr3.set_title("ZCA White Covariance\n")
	plt.tight_layout()
	plt.show()
	

# -----------------PCA and ZCA whiten digits--------------
	# Plot 10 digits after PCA whitening
	# 1 example of every digit
	
	fig, ax = plt.subplots(2, 5, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                      gridspec_kw=dict(hspace=0.1, wspace=0.1))



	for i in range(5):
	    ax[0, i].imshow(PCAwhite[:,100*i].reshape(10, 10), cmap='jet')
	
	for i in range(5):
	    ax[1, i].imshow(PCAwhite[:,100*i+500].reshape(10, 10), cmap='jet')
	   
	fig.suptitle("PCA White Digits")
	plt.show()


	fig, ax = plt.subplots(2, 5, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))

	for i in range(5):
	    ax[0, i].imshow(ZCAWhite[:,100*i].reshape(28, 28), cmap='jet')
	
	for i in range(5):
	    ax[1, i].imshow(ZCAWhite[:,100*i+500].reshape(28, 28), cmap='jet')
	   

	fig.suptitle("ZCA White Digits")
	plt.show()

	# reduced data
	return xTilde


def FLD(data):
	# get data for 0 and 1 digits

	# 0 digits - 100*100
	data1 = data[:,1:101]
	# 1 digits
	data2 = data[:,101:201]

# ---------------- 0 digit -- class 1 
	ImageVector1=np.asarray(data1,dtype=float)												 #convert list to np.array 
	
	m1=np.mean(ImageVector1,axis=1)								 #mean about columns (axis=1 gives mean about column)

	A1=ImageVector1-m1[:,None]    							         #Difference matrix(Unique features)
	At1=A1.T 														 #Transpose of Difference matrix				
	covar1=A1.dot(At1)												 #covariance = A transpose into A

# ---------1 digit --- class 2
	ImageVector2=np.asarray(data2,dtype=float)												 #convert list to np.array 
	
	m2=np.mean(ImageVector2,axis=1)								 #mean about columns (axis=1 gives mean about column)

	A2=ImageVector2-m2[:,None]    							         #Difference matrix(Unique features)
	At2=A2.T 														 #Transpose of Difference matrix				
	covar2=A2.dot(At2)												 #covariance = A transpose into A

	Sw = covar1+covar2

	diffMean = (m1 - m2)
	# print Sw.shape
	SwInv = linalg.inv(Sw)

	W = SwInv.dot(diffMean)
	
	# transform data to this direction
	Wt = W.T
	projectedMean1 = Wt.dot(m1)
	projectedMean2 = Wt.dot(m2)

	print "Projected mean 1:", projectedMean1,"\nProjected Mean 2: ",projectedMean2
	# print Wt.shape, ImageVector1.shape
	data1InW  = Wt.dot(ImageVector1)
	data2InW  = Wt.dot(ImageVector2)
	# print data1InW,data2InW
	
	threshold = (projectedMean1+projectedMean2)/2

	c1=0
	c2=0
	completeData =[]
	completeData.extend(data1InW)
	completeData.extend(data2InW)
	
	# print len(completeData)
	for i in completeData:
		if(i<=threshold):
			c1+=1
		else:
			c2+=1

	# print c1,c2
	print "Threshold for classification:", threshold
	return W,threshold


def testFLD(testData,W,threshold):
	# 100*10 dimension  100 features 10 images
	data1 = testData[:,0:10]
	data2 = testData[:,10:20]
	# print data1.shape
	Wt = W.T
	data1InW  = Wt.dot(data1)
	data2InW  = Wt.dot(data2)
	# print data1InW,data2InW
	counterror=0
	for i in data1InW:
		if(i<=threshold):
			counterror+=1;

	for i in data2InW:
		if(i>threshold):
			counterror+=1;

	# print counterror	
	# print data1.shape,data2.shape	
	error  = float(counterror)/float(data2InW.size+data2InW.size)*100
	print "Accuracy For Fisher's Linear Discriminant: ", 100-error,"%"

def main():
	trX, trY, tsX, tsY = mnist(noTrSamples=1000,
                               noTsSamples=100, digit_range=[0, 10],
                               noTrPerClass=100, noTsPerClass=10)

	#------------------ORIGINAL IMAGES------------------


	fig, ax = plt.subplots(2, 5, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
	fig.suptitle("Original Data")
	for i in range(5):
		# print 100*i
		ax[0, i].imshow(trX[:,100*i].reshape(28, 28), cmap='jet')
	
	for i in range(5):
		# print 100*i
		ax[1, i].imshow(trX[:,100*i+500].reshape(28, 28), cmap='jet')
	
	plt.show()
	reducedTrainData,U,S,mean =  Reduce(trX)
	plotData(reducedTrainData,U,S)
												 
	# print ImageVector.shape
	#----------------Eigen_faces------------------------------------

	# print tsX.shape
	sort_indices = np.argsort(tsY[0])
	# print sort_indices
	sorted_labels = tsY[0][sort_indices]
	labels = sorted_labels[:20]
	# print tsX.shape
	sorted_test_data = tsX[:,sort_indices]
	test_data = sorted_test_data[:,:20]
	# print np.argsort(tsYlist)

	# print "gi",test_data.shape
	# print test_data.shape
	ImageVector=np.asarray(test_data)
	# print ImageVector.shape

	A=ImageVector-mean[:,None]  
	# A - 100*20
	
	reducedTestData = U[:,1:101].T.dot(A)	
	W,threshold = FLD(reducedTrainData)
	testFLD(reducedTestData,W,threshold)


if __name__ == "__main__":
    main()
