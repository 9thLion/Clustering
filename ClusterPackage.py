import numpy as np 
import random as ran
import matplotlib.pyplot as plt


def Kmeans(X, k=2,distance='euclidean', maxiter=10000, reps=100): 
	print("Now runnning Kmeans..")
	#Define the kmean algorithm:
	def Kmean(X):
		#Based on chosen distance function:
		def Dist(x,y):
			if distance == 'euclidean':
				return ((sum((x-y)**2))**(1/2))
			if distance == 'mahalanobis':
				Si=np.linalg.inv(X.dot(X.T))
				return (((x-y).T.dot(Si).dot(x-y))**(1/2))
			if distance == 'manhattan':
				return (sum(abs(x-y)))
		centroids=[]
		#Take some random data points as centroids like so:
		for x in range(k): 
			centroids.append(ran.choice(X.T))
		#The case of randomly picking centroids that are close is an issue that should be adressed.
		centroids = np.array(centroids)
		Iter=0 #Initialize iterations
		while True:
			Iter+=1
			Clusters = {x:[] for x in range(k)} 
			Distance = {x:[] for x in range(k)} #Save that to choose the best replication
			#the index of the centroid within the centroids array will match the key
			Color=[]
			for x in X.T: #data points
				centroid = np.array(centroids[0]) #starting with the first
				#In each loop the euclidean distance between the data point and each centroid
				#is calculated. Then the data point is matched to the centroid with the least distance.
				#So initialize with centroid indexed 0 and then loop for the rest. 
				Eucprevious = Dist(x,centroid)
				counter = index = 0 
				#The counter at the end of each loop will match the index of the centroid
				for c in centroids[1:]:
					c = np.array(c)
					if Dist(x,c) < Eucprevious:
						Eucprevious = Dist(x,c)
						centroid = c
						counter += 1 
						#counter and index need to be separate variables, 
						#as counter keeps on counting even
						#after the optimal centroid was found
						index = counter
					else:
						counter+=1
				#Construct the final output, a label vector to visualize the clusters defined
				#The index will do the job
				Color.append(index)
				#Also a dictionary to were the cluster label is matched with the data points
				Clusters[index].append(x)
				Distance[index].append(Eucprevious)

			centroids_new=[] #initialize the new centroids

			#The case of bad seeds causes unexpected errors occasionally, it needs to be handled.
			try:
				for key in Clusters:
					#stack the values of each cluster as columns of a new array
					OldCluster = np.column_stack(Clusters[key])
					#and now compute the new centroids
					centroids_new.append(np.mean(OldCluster,1))
				centroids_new = np.array(centroids_new)
			except ValueError: 
				print("A replication failed due to bad seeds")
				Colors = 0
				centroids = 0
				TotalDist = 10000000000000
				break

			metric=[Dist(centroids_new[i], centroids[i]) for i in range(len(centroids))]
			if sum(metric)/len(metric) < 0.0000001: #calculating the mean of the distance between all old with new centroids
				#Keep the metric, after the many replications of Kmean
				#we will use it to find the best centroids
				Flattened=[item for sublist in Distance.values() for item in sublist]
				TotalDist = sum(Flattened)
				break

			elif Iter == maxiter:
				Flattened=[item for sublist in Distance.values() for item in sublist]
				TotalDist = sum(Flattened) #total distance within clusters
				print("k-means Convergence failed after", maxiter, "runs")
				break
			else:
				centroids = centroids_new
		return(Color, centroids, TotalDist)
	#Now for multiple repetitions:
	Metric = 100000000 #just choose a big initial number
	colors = []
	for x in range(reps):
		col, cen, met = Kmean(X)
		#Select the lowest total distance replication 
		if met < Metric:
			Metric = met
			finalcentroids = cen
			LabelVector = np.array(col)
	return(finalcentroids, LabelVector)	

def MoG(X, K=2, maxiter=1000, reps=10):
	print("Now running Mixture of Gaussians...")
	D=X.shape[0]
	print("For each initialization, Kmeans algorithm will also be run to calculate the initial means")
	def OneMoG(X):
		#Parameter initialization#
		#Initial means will be the output centroids
		means, label = Kmeans(X,k=K, reps=25) 
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

		Iter = 0 #initialize Iterations
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
			mlike_new = maxlikelihood(X, w=weights_new, m=means_new, s=sigma_new)
			if abs(mlike_new - mlike)<0.000001: #10^6
				labels=[]

				for x in range(rs.shape[0]):
					for y in rs[x]:
						if y!=np.amax(rs[x]):
							rs[rs==y]=0
				rs[rs!=0]=1
				#latent variable uncovered.

				for x in range(rs.shape[0]):
					for y in range(rs.shape[1]):
						if rs[x][y] == 1:
							labels.append(y)
				labels = np.array(labels)

				distances=[]
				for x in X.T:
					euclidean = 1000000
					for m in means_new:
						euclidean_new = (sum((x-m)**2))**(1/2)
						if euclidean_new < euclidean:
							euclidean = euclidean_new
					distances.append(euclidean)
				TotalDist = sum(distances)
				break
			else:
				weights = weights_new
				means = means_new
				sigma = sigma_new
				mlike = mlike_new
			Iter+=1
			if Iter == maxiter:
				print('MoG failed to converge after', Iter, 'iterations')
				labels=[]

				for x in range(rs.shape[0]):
					for y in rs[x]:
						if y!=np.amax(rs[x]):
							rs[rs==y]=0
				rs[rs!=0]=1
				#latent variable uncovered.

				for x in range(rs.shape[0]):
					for y in range(rs.shape[1]):
						if rs[x][y] == 1:
							labels.append(y)
				labels = np.array(labels)

				distances=[]
				for x in X.T:
					euclidean = 1000000
					for m in means_new:
						euclidean_new = (sum((x-m)**2))**(1/2)
						if euclidean_new < euclidean:
							euclidean = euclidean_new
					distances.append(euclidean)
				TotalDist = sum(distances)
				break

		return(means_new, labels, TotalDist)
	totalDist = 1000000
	for x in range(reps):
		print("Replication number", x)
		means, labels, totalDist_new = OneMoG(X)
		if totalDist_new < totalDist:
			totalDist = totalDist_new
			Means = means
			Labels = labels
	return(Means, Labels)


#This silhouette function doesn't produce correct results
#It took me some time and i couldn't figure out why, so i ended up using the built-in
#instead. I would like some feedback on why this doesn't work though.
def Silhouette(Data, labels):
	X=Data.T #Data input will be DxN
	def euc(x,y):
		return ((sum((x-y)**2))**(1/2))
	S=[]
	for i in set(labels):
		s=[]
		#for every data point with a specific label (for every data point within a specific cluster)
		for x in X[labels==i]:
			#First find the distance within the cluster
			temp=[]
			for y in X[labels==i]:
				if euc(x,y)==0: #skip when basically x is y
					continue
				temp.append(euc(x,y))
			a=sum(temp)/len(temp)

			#Then find the distance from each of the other clusters
			B=[]
			for j in set(labels):
				if j == i:
					continue

				temp=[]
				for y in X[labels==j]:
					temp.append(euc(x,y))
				B.append((sum(temp)/len(temp)))
			b=float(min(B))
			s.append((b-a)/max(a,b))

		#one list for each cluster, appended to another list
		S.append(s)

	temp=[]
	for l in S:
		temp.append(sum(l)/len(l)) #silhouette coefficient for each cluster
	Sil=sum(temp)/len(temp) #total silhouette coefficient
	return(Sil)

