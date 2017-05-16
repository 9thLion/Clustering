#! usr/bin/python3.5

import numpy as np
from numpy.linalg import inv
from scipy import linalg


def MyFirstAlgorithm(X,mode='cov',k=1, threshold=0.3): 
	print('Now running conventional PCA through eigendecomposition of the covariance matrix...')
	if mode == 'cov':
		#Calculation of Covariance matrix
		C = np.cov(X) 
	elif mode == 'noncov':
		#Calculation of the Scatter matrix
		meanM=[]
		for line in X:
			meanM.append([np.mean(line)])
		meanMatrix=np.array(meanM)
		C = np.array(sum([np.mat(x-meanMatrix).T.dot(np.mat(x-meanMatrix)) for x in X.T]))
	eigVals,eigVecs = linalg.eigh(C) #the eigenvalues are given in ascending order
	index=eigVals.argsort()[::-1]  #finding the indexes for descending order
	eigvecs=eigVecs[:k,index] #the first k vectors will be kept
	W=eigvecs[:]
	y=W.dot(X)
	return(y, eigVals)    

def PCA(Data, k=1, F=True, threshold=0.02): 
	print('Now running conventional PCA through singular value decomposition of the data...')
	D = Data.shape[0]
	N = Data.shape[1]
	meanM = []
	for line in Data:
		meanM.append([np.mean(line)])
	meanMatrix = np.array([meanM]*N).squeeze().T
	X = Data - meanMatrix #centered	

	U,S,V = np.linalg.svd(X, full_matrices=F) 
	eigvecs=U.T[:k] #the first k vectors will be kept
	W=eigvecs[:]
	y=W.dot(Data)
	return(y, S)
	
def PPCA(Data, M=1, variance=True): 
	pie=np.pi
	D = Data.shape[0]
	N = Data.shape[1]
	meanM = []
	for line in Data:
		meanM.append([np.mean(line)])
	meanMatrix = np.array([meanM]*N).squeeze().T
	X = Data - meanMatrix #centered	


	if variance==False: #Sigma is zero
		print('Now running PPCA (sigma zero) on the data...')
		#SET ARBITRARY W#
		mean = np.zeros(M)
		var = np.identity(M)
		W = np.random.multivariate_normal(mean,var,D) #DxM matrix
		while True:
			Mi = inv(W.T.dot(W))  
			O = Mi.dot(W.T).dot(X) #O is the expectation E[x] where x are the observed values
			
			W_new = X.dot(O.T).dot(inv(O.dot(O.T)))
			
			if sum(sum(abs(W_new-W)**2)) < 0.000001:
				U,S,V = np.linalg.svd(W_new, full_matrices=False)
				Output = U.T.dot(Data)
				break
			else:
				W = W_new
		return (Output)

	if variance==True: #Sigma is not zero
		print('Now running PPCA (sigma not zero) on the data...')
		Runs=0
		Wfinal=np.zeros((D,M))
		def EM(X):
			#SET ARBITRARY W#
			W = np.random.rand(D,M)
			#SET ARBITRARY Sigma#
			sigma = np.random.random()
			metric = 0
			counter = 0
			while True:
				#In the following loops i will split the terms of W_new and sigma_new to separate variables to avoid a huge equation.
				counter+=1

				Mi = inv(W.T.dot(W) + sigma*np.identity(W.shape[1]))
				Ez = []
				Ezzt = []
				A = np.zeros((D,M)) #Starting with a zero matrix, to sum the first term of Wnew
				

				for x in X.T:
					x = np.mat(x)
					Ez_n = Mi.dot(W.T).dot(x.T)
					Ezzt_n = sigma*Mi + Ez_n.dot(Ez_n.T)
					Ezzt.append(Ezzt_n)
					A += x.T.dot(Ez_n.T)

				Ezzt = np.array(Ezzt) #Nx(MxM)
				B = inv(sum(Ezzt)) #second term of Wnew
				W_new = A.dot(B)

				values=[] 
				for x in X.T:
					x = np.mat(x)
					Ez_n = Mi.dot(W.T).dot(x.T)
					a = x.dot(x.T)
					b = Ez_n.T.dot(W_new.T).dot(x.T)
					Ezzt_n = sigma*Mi + Ez_n.dot(Ez_n.T)
					c = np.trace(Ezzt_n.dot(W_new.T).dot(W_new))
					values.append(float(a - 2*b + c))

				sigma_new = (1/N*D)*sum(values)

				#We will use the joint p(x,z) expectation given the parameters as convergence metric
				#In each run we will compare with the previous
				#In the first run we have defined the old metric as 0 to skip right ahead.
				a = (D/2)*np.log(2*pie*sigma_new)
				b = []
				c = []
				d = []
				e = []
				for x in X.T:
					x = np.mat(x)
					Ez_n = Mi.dot(W.T).dot(x.T)
					Ezzt_n = sigma*Mi + Ez_n.dot(Ez_n.T)
					b.append(float(1/2)*np.trace(Ezzt_n))
					c.append(float((1/2*sigma_new)*(x.dot(x.T))))
					d.append(float((1/sigma_new)*Ez_n.T.dot(W_new.T).dot(x.T)))
					e.append(float((1/2*sigma_new)*np.trace(Ezzt_n.dot(W_new.T).dot(W_new))))

				metric_new = -sum(a+b+c-d+e)
				print('Maximization number',str(counter),'expectation metric:',metric_new)

				if abs(metric_new-metric)<0.00000001: #10^-8
					print('Converged at ', counter, '!')
					break
				else:
					W = W_new
					sigma = sigma_new
					metric = metric_new
			return W_new

		EMruns=10
		for x in range(EMruns):
			Wfinal+=EM(X)
		Wfinal=Wfinal/EMruns #Running EM multiple times and then keeping the mean Matrix of all the outputs
		U,S,V=np.linalg.svd(Wfinal, full_matrices=False)
		Output=U.T.dot(Data)
		return(Output)
	

		
def KPCA(X, gamma=3, dims=1, mode='gaussian'): 
	print('Now running Kernel PCA with', mode, 'kernel function...')
	'''
	X is the necessary input. The data.
	gamma will be the user defined value that will be used in the kernel functions. The default is 3.
	dims will be the number of dimensions of the final output (basically the number of components to be picked). The default is 1.
	mode has three options 'gaussian', 'polynomial', 'hyperbolic tangent' which will be the kernel function to be used. The default is gaussian.
	'''

	#First the kernel function picked by the user is defined. Vectors need to be input in np.mat type
	
	def phi(x1,x2):   
		if mode == 'gaussian':
			return (float(np.exp(-gamma*((x1-x2).dot((x1-x2).T))))) #gaussian. (vectors are rather inconvenient in python, so instead of xTx for inner product we need to calculate xxT)
		if mode == 'polynomial':
			return (float((1 + x1.dot(x2.T))**gamma)) #polynomial
		if mode == 'hyperbolic tangent':
			return (float(np.tanh(x1.dot(x2.T) + gamma))) #hyperbolic tangent
	Kernel=[]
	for x in X.T:
		xi=np.mat(x)
		row=[]
		for y in X.T:
			xj=np.mat(y)
			kf=phi(xi,xj)
			row.append(kf)
		Kernel.append(row)
	kernel=np.array(Kernel)

	# Centering the symmetric NxN kernel matrix.
	N = kernel.shape[0]
	one_n = np.ones((N,N)) / N
	kernel = kernel - one_n.dot(kernel) - kernel.dot(one_n) + one_n.dot(kernel).dot(one_n) #centering

	eigVals, eigVecs = linalg.eigh(kernel) #the eigvecs are sorted in ascending eigenvalue order.
	y=eigVecs[:,-dims:].T #user defined dims, since the order is reversed, we pick principal components from the last columns instead of the first
	return (y)
