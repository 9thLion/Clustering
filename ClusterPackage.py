import numpy as np 
import random as ran
import matplotlib.pyplot as plt


def Kmeans(Data, k=2,distance='euclidean', maxiter=25, reps=10): 
#reps should not be too small, it could cause a problem in the output
	N = Data.shape[1]
	meanM=[]
	for line in Data:
		meanM.append([np.mean(line)])
	meanMatrix = np.array([meanM]*N).squeeze().T
	X = Data - meanMatrix 

	def Kmean(X):
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
		centroids = np.array(centroids)
		Iter=0 #first iteration
		while True:
			Iter+=1
			Clusters = {x:[] for x in range(k)} #the index of the centroid within the centroids array will match the key
			Color=[]
			for x in X.T: #data points
				centroid = np.array(centroids[0]) #starting with the first
				Eucprevious = Dist(x,centroid)
				counter=index=0 #The counter at the end of each loop will match the index of the centroid
				for c in centroids[1:]:
					c = np.array(c)
					if Dist(x,c) < Eucprevious:
						Eucprevious = Dist(x,c)
						centroid = c
						counter += 1 #first add one, so that the counter will be the same as the index
						index = counter 
					else:
						counter+=1
				Color.append(index) #The final output, to visualize the clusters
				Clusters[index].append(x)

			centroids_new=[] #empty it
			for key in Clusters:
				#stack the values of each cluster as columns of a new array
				OldCluster = np.column_stack(Clusters[key])
				#and now compute the new centroids
				centroids_new.append(np.mean(OldCluster,1))
			centroids_new = np.array(centroids_new)
			metric=[Dist(centroids_new[i], centroids[i]) for i in range(len(centroids))]
			if sum(metric)/len(metric) < 0.0000001: #calculating the mean of the distance between all old with new centroids
				break
			elif Iter == maxiter:
				print("Convergence failed after", maxiter, "runs")
				break
			else:
				print('One more loop done and the new centroids are', centroids_new)
				centroids = centroids_new
		return(Color, centroids)
	#Now for multiple repetitions:
	finalcentroids  = np.zeros((k,X.shape[0]))
	colors = []
	for x in range(reps):
		col, cen = Kmean(X) #I'll just save the last centroid
		colors.append(col)
		#Is there a way to compare the right centroids and keep their mean? they always switch.
		#finalcentroids += cen
	#print(finalcentroids)
	#OutputCentroids = finalcentroids / reps
	C = np.column_stack([np.array(x) for x in colors])
	ColorVector = []
	#For the label vector we find the most frequent elements among all runs:
	for c in C:
		counts = np.bincount(c)
		ColorVector.append(np.argmax(counts))
	ColorVector = np.array(ColorVector)
	return(ColorVector, cen)	
