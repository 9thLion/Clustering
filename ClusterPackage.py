import numpy as np 
import random as ran
import matplotlib.pyplot as plt


def Kmeans(Data, k=2,distance='euclidean', maxiter=25, reps=100): 
#reps should not be too small, it could cause a problem in the output

	#Pre-Processing:
	N = Data.shape[1]
	meanM=[]
	for line in Data:
		meanM.append([np.mean(line)])
	meanMatrix = np.array([meanM]*N).squeeze().T
	X = Data - meanMatrix 

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

			centroids_new=[] #initialize the new centroids
			for key in Clusters:
				#stack the values of each cluster as columns of a new array
				OldCluster = np.column_stack(Clusters[key])
				#and now compute the new centroids
				centroids_new.append(np.mean(OldCluster,1))
			centroids_new = np.array(centroids_new)
			metric=[Dist(centroids_new[i], centroids[i]) for i in range(len(centroids))]
			if sum(metric)/len(metric) < 0.0000001: #calculating the mean of the distance between all old with new centroids
				#Keep the metric, after the many replications of Kmean
				#we will use it to find the best centroids
				Metric = sum(metric)/len(metric)
				break
			elif Iter == maxiter:
				print("Convergence failed after", maxiter, "runs")
				break
			else:
				print('Iteration', Iter, 'done. The new centroids are:', centroids_new)
				centroids = centroids_new
		return(Color, centroids, Metric)
	#Now for multiple repetitions:
	Metric = 1 #just choose a big initial number, everything will be lower than 10^7 anyway.
	colors = []
	for x in range(reps):
		col, cen, met = Kmean(X) #I'll just save the last centroid
		if met < Metric:
			Metric = met
			finalcentroids = cen
		colors.append(col)
	#stack the label vectors as columns of a matrix:
	C = np.column_stack([np.array(x) for x in colors]) 

	ColorVector = []
	#then find the most frequent elements among all runs:
	for c in C:
		counts = np.bincount(c)
		ColorVector.append(np.argmax(counts))
	ColorVector = np.array(ColorVector)
	return(ColorVector, finalcentroids)	
