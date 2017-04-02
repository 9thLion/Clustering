#! user/bin/python3.5

import numpy as np 
import random as ran
import matplotlib.pyplot as plt
import ClusterPackage as clu

#Crafting the Artificial Dataset
D=2
N1=220
N2=280

mean = np.ones(D)
def varID(scale):
	return(scale*np.identity(D)) 
X1 = np.random.multivariate_normal(mean, varID(0.5), N1).T #DxN matrix
X2 = np.random.multivariate_normal(-mean, varID(0.75), N2).T
X = np.concatenate((X1,X2), axis=1)


K=1
D=2
#Parameter initialization#
#Initial means will be the output centroids
label, means = clu.Kmeans(X,k=K, reps=5, quiet=True) 
print(label)
#Initial weights
weights = np.array([1/K for x in range(K)])
#Initial covariance
sigma = np.ones((K,D,D))
for x in range(K):
	#Transpose the matrix to pick the samples with the same label x and transpose back to calculate the covariance
	#the means index will match the index of the covariance matrix, because the index of the centroids match the labels
	sigma[x]=np.cov((X.T[label==x]).T)

#define the pdf of multivariate gaussian:
def gaussian(x, mu, sig, d):
	return (1/np.sqrt(((2*np.pi)**d)*np.linalg.det(sig)))*np.exp(-(1/2)*(x-mu).T.dot(np.linalg.inv(sig)).dot(x-mu))

#up next, define the likelihood function and compute the initial likelihood
def maxlikelihood(X):
	outer = []
	for x in X.T:
		inner = []
		for i in range(K):
			inner.append(weights[i]*gaussian(x, means[i], sigma[i], d=D))
		outer.append(sum(inner))	
	maxlikelihood = sum(np.log(outer))
	return maxlikelihood

mlike = maxlikelihood(X)

#Expectation Step, responsibility calculation
rs = []
for x in X.T:
	inner = []
	for i in range(K):
		inner.append(weights[i]*gaussian(x, means[i], sigma[i], d=D))
	rs.append(inner/sum(inner))
rs = np.array(rs)
print(rs)

#Maximization Step
N = sum(rs)

inner = []
for x in X.T:
	inner.append(rs*x)
print(inner[1])
means_new = (1/N) * sum(inner)


inner = []
rs = np.mat(rs)
for x in X.T:
	x = np.mat(x) #pote ksana python
	inner.append(np.mat(x-means_new).T*np.mat(x-means_new))

