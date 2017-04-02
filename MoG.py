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

def MoG(X, K=2):
	D=X.shape[0]
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
	def maxlikelihood(X,w,m,s):
		outer = []
		for x in X.T:
			inner = []
			for i in range(K):
				inner.append(w[i]*gaussian(x, m[i], s[i], d=D))
			outer.append(sum(inner))	
		maxlikelihood = sum(np.log(outer))
		return maxlikelihood
	mlike = maxlikelihood(X, w=weights, m=means, s=sigma)

	while True:
		#Expectation Step, responsibility calculation
		rs = []
		for x in X.T:
			inner = []
			for i in range(K):
				inner.append(weights[i]*gaussian(x, means[i], sigma[i], d=D))
			rs.append(inner/sum(inner))
		rs = np.array(rs)
		#Maximization Step, 3 parameters
		N = sum(rs)
		print(N[0])
		means_new = []
		for k in range(K):
			inner = []
			for n in range(X.shape[1]): #that's the number of columns, which means the number of samples
				inner.append(rs[n][k]*X.T[n])
			means_new.append((1/N[k]) * sum(inner))
		means_new = np.array(means_new)

		X=np.mat(X) #to make vector calculations
		sigma_new = []
		for k in range(K):
			inner = []
			for n in range(X.shape[1]):
				#Dont forget, default vectors in python are horizontal, so transpose the other way around
				inner.append(rs[n][k]*np.mat(X.T[n]-means_new[k]).T*np.mat(X.T[n]-means_new[k]))
			sigma_new.append((1/N[k]) * sum(inner))
		sigma_new = np.array(sigma_new)
		X=np.array(X) #back to array


		weights_new = []
		for k in range(K):
			weights_new.append(N[k]/sum(N))
		weights_new = np.array(weights_new)
		print(rs)
		mlike_new = maxlikelihood(X, w=weights_new, m=means_new, s=sigma_new)
		if abs(mlike_new - mlike)<0.000001:
			
			break
		else:
			weights = weights_new
			means = means_new
			sigma = sigma_new
			mlike = mlike_new
	return(means_new)

print(MoG(X))
